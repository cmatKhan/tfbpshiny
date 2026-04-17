locals {
  log_group_app     = "/tfbpshiny/production"
  log_group_traefik = "/tfbpshiny/production/traefik"
  namespace         = "TFBPShiny"
}

# ---------------------------------------------------------------------------
# Log groups
# ---------------------------------------------------------------------------
# Declaring these explicitly so Terraform owns retention/lifecycle. If either
# already exists in the account, import it with:
#   terraform import aws_cloudwatch_log_group.app /tfbpshiny/production
#   terraform import aws_cloudwatch_log_group.traefik /tfbpshiny/production/traefik
#
# NOTE: After applying, update the Docker awslogs driver for the Traefik
# container to use awslogs-group=/tfbpshiny/production/traefik so its noisy
# network-discovery warnings stop polluting the Shiny app's log group.

resource "aws_cloudwatch_log_group" "app" {
  name              = local.log_group_app
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "traefik" {
  name              = local.log_group_traefik
  retention_in_days = 14
}

# ---------------------------------------------------------------------------
# Metric filters — extract numeric values from pipe-delimited log records
# ---------------------------------------------------------------------------
# Bracket-pattern filters tokenize on whitespace, so every pipe character is
# its own token (p1..p7). Field positions below reflect that.
#
# Record format emitted by the app:
#   PROFILE | <ts> | <elapsed_s> | <op> | <module> | <dataset> | <context> | <sid>
#   SESSION | <ts> | START|END   | <sid>
#
# default_value is intentionally omitted. If it were set to 0, every minute
# without a PROFILE event would emit a zero, dragging the latency average
# toward zero and suppressing the alarm during low traffic.

# All PROFILE records — used for dashboard widgets that show latency across
# every operation. Not used for the vdb.query-specific alarm.
resource "aws_cloudwatch_log_metric_filter" "profile_latency" {
  name           = "tfbpshiny-profile-latency"
  log_group_name = aws_cloudwatch_log_group.app.name
  pattern        = "[marker=PROFILE, p1, ts, p2, elapsed_s, p3, op, p4, module, p5, dataset, p6, context, p7, sid]"

  metric_transformation {
    name      = "ProfileLatencySeconds"
    namespace = local.namespace
    value     = "$elapsed_s"
    unit      = "Seconds"
  }
}

# Narrow filter that only matches op=vdb.query. Drives the latency alarm.
resource "aws_cloudwatch_log_metric_filter" "vdb_query_latency" {
  name           = "tfbpshiny-vdb-query-latency"
  log_group_name = aws_cloudwatch_log_group.app.name
  pattern        = "[marker=PROFILE, p1, ts, p2, elapsed_s, p3, op=vdb.query, p4, module, p5, dataset, p6, context, p7, sid]"

  metric_transformation {
    name      = "VdbQueryLatencySeconds"
    namespace = local.namespace
    value     = "$elapsed_s"
    unit      = "Seconds"
  }
}

# Counts SESSION START records — one per new user connection.
resource "aws_cloudwatch_log_metric_filter" "session_start" {
  name           = "tfbpshiny-session-start"
  log_group_name = aws_cloudwatch_log_group.app.name
  pattern        = "[marker=SESSION, p1, ts, p2, event=START, p3, sid]"

  metric_transformation {
    name          = "SessionStart"
    namespace     = local.namespace
    value         = "1"
    default_value = 0
    unit          = "Count"
  }
}

# ---------------------------------------------------------------------------
# Alarms
# ---------------------------------------------------------------------------
# Fires when avg vdb.query latency exceeds 5 s over 15 min (3 × 5-min periods).
# Raise the threshold to 10–30 s while the app is known to be slow; drop it
# toward 2–5 s once query performance improves.
resource "aws_cloudwatch_metric_alarm" "high_vdb_query_latency" {
  alarm_name          = "tfbpshiny-high-vdb-query-latency"
  alarm_description   = "Avg vdb.query latency > 5 s over 15 min — investigate vector DB bottlenecks."
  namespace           = local.namespace
  metric_name         = "VdbQueryLatencySeconds"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 3
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
}

# Fires when more than 20 sessions start within a 5-minute window.
# Adjust threshold based on observed normal traffic once the app is live.
resource "aws_cloudwatch_metric_alarm" "high_session_count" {
  alarm_name          = "tfbpshiny-high-session-count"
  alarm_description   = "More than 20 new sessions in 5 min — unexpected traffic spike."
  namespace           = local.namespace
  metric_name         = "SessionStart"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 20
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
}

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
# All Logs Insights queries target only the app log group (Traefik lives
# elsewhere now), so the PROFILE/SESSION filters aren't competing with
# reverse-proxy noise. Parse patterns use the exact pipe-with-spaces format
# the app emits; if you ever change that format, update these in lockstep.

resource "aws_cloudwatch_dashboard" "tfbpshiny" {
  dashboard_name = "tfbpshiny-production"
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "log"
        x      = 0
        y      = 0
        width  = 24
        height = 6
        properties = {
          title         = "Avg and max latency by operation"
          region        = var.aws_region
          view          = "table"
          logGroupNames = [local.log_group_app]
          queryString   = <<-EOT
            fields @timestamp, @message
            | filter @message like /^PROFILE \|/
            | parse @message "PROFILE | * | * | * | * | * | * | *" as ts, elapsed_s, op, module, dataset, context, sid
            | stats avg(elapsed_s) as avg_s, max(elapsed_s) as max_s, count() as n by op, module, context
            | sort avg_s desc
          EOT
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title         = "Sessions started (5-min bins)"
          region        = var.aws_region
          view          = "timeSeries"
          logGroupNames = [local.log_group_app]
          queryString   = <<-EOT
            fields @timestamp, @message
            | filter @message like /^SESSION \|/
            | parse @message "SESSION | * | * | *" as ts, event, sid
            | filter event = "START"
            | stats count(*) as sessions_started by bin(5m)
            | sort @timestamp asc
          EOT
        }
      },
      {
        type   = "log"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title         = "Unique visitors per day"
          region        = var.aws_region
          view          = "bar"
          logGroupNames = [local.log_group_app]
          queryString   = <<-EOT
            fields @timestamp, @message
            | filter @message like /^SESSION \|/
            | parse @message "SESSION | * | * | *" as ts, event, sid
            | filter event = "START"
            | stats count_distinct(sid) as unique_visitors by bin(1d)
            | sort @timestamp asc
          EOT
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 12
        width  = 24
        height = 6
        properties = {
          title         = "Latency by operation over time (5-min bins)"
          region        = var.aws_region
          view          = "timeSeries"
          logGroupNames = [local.log_group_app]
          queryString   = <<-EOT
            fields @timestamp, @message
            | filter @message like /^PROFILE \|/
            | parse @message "PROFILE | * | * | * | * | * | * | *" as ts, elapsed_s, op, module, dataset, context, sid
            | stats avg(elapsed_s) as avg_s by bin(5m), op
            | sort @timestamp asc
          EOT
        }
      },
      {
        type   = "alarm"
        x      = 0
        y      = 18
        width  = 12
        height = 3
        properties = {
          title  = "vdb.query latency alarm"
          alarms = [aws_cloudwatch_metric_alarm.high_vdb_query_latency.arn]
        }
      },
      {
        type   = "alarm"
        x      = 12
        y      = 18
        width  = 12
        height = 3
        properties = {
          title  = "High session count alarm"
          alarms = [aws_cloudwatch_metric_alarm.high_session_count.arn]
        }
      },
    ]
  })
}
