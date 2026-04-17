# NOTE! THIS isn't working to create the dashboard. But I could paste in queries
# by hand to each panel and that did work so the logging is producing correct
# values

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

resource "aws_cloudwatch_log_group" "app" {
  name              = local.log_group_app
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "traefik" {
  name              = local.log_group_traefik
  retention_in_days = 14
}

# ---------------------------------------------------------------------------
# Metric filters — JSON format, no tokenization ambiguity
# ---------------------------------------------------------------------------
# Records emitted by the app are single-line JSON objects:
#   PROFILE: {"event":"PROFILE","ts":"...","elapsed_s":0.1234,"op":"...","module":"...","dataset":"...","context":"...","session_id":"..."}
#   SESSION: {"event":"SESSION","ts":"...","lifecycle":"START"|"END","session_id":"..."}
#
# CloudWatch JSON metric filters match on field values and extract numeric
# fields directly, with no whitespace-tokenization issues.
#
# default_value is intentionally omitted on latency filters. Setting it to 0
# would drag the average toward zero during quiet periods and suppress alarms.

resource "aws_cloudwatch_log_metric_filter" "profile_latency" {
  name           = "tfbpshiny-profile-latency"
  log_group_name = aws_cloudwatch_log_group.app.name
  pattern        = "{ $.event = \"PROFILE\" }"

  metric_transformation {
    name      = "ProfileLatencySeconds"
    namespace = local.namespace
    value     = "$.elapsed_s"
    unit      = "Seconds"
  }
}

# Narrow filter for vdb.query only — drives the latency alarm.
resource "aws_cloudwatch_log_metric_filter" "vdb_query_latency" {
  name           = "tfbpshiny-vdb-query-latency"
  log_group_name = aws_cloudwatch_log_group.app.name
  pattern        = "{ $.event = \"PROFILE\" && $.op = \"vdb.query\" }"

  metric_transformation {
    name      = "VdbQueryLatencySeconds"
    namespace = local.namespace
    value     = "$.elapsed_s"
    unit      = "Seconds"
  }
}

# Counts SESSION START records — one per new user connection.
resource "aws_cloudwatch_log_metric_filter" "session_start" {
  name           = "tfbpshiny-session-start"
  log_group_name = aws_cloudwatch_log_group.app.name
  pattern        = "{ $.event = \"SESSION\" && $.lifecycle = \"START\" }"

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
# Raise threshold while the app is known to be slow; lower toward 2-5 s
# once query performance improves.
resource "aws_cloudwatch_metric_alarm" "high_vdb_query_latency" {
  alarm_name          = "tfbpshiny-high-vdb-query-latency"
  alarm_description   = "Avg vdb.query latency > 30 s over 15 min — investigate query bottlenecks."
  namespace           = local.namespace
  metric_name         = "VdbQueryLatencySeconds"
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 3
  threshold           = 30
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
}

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
# Logs Insights auto-discovers JSON fields, so no parse directive needed.
# Queries filter on $.event to separate PROFILE from SESSION records.

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
            fields @timestamp, op, module, dataset, context, elapsed_s
            | filter event = "PROFILE"
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
            fields @timestamp, session_id
            | filter event = "SESSION" and lifecycle = "START"
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
            fields @timestamp, session_id
            | filter event = "SESSION" and lifecycle = "START"
            | stats count_distinct(session_id) as unique_visitors by bin(1d)
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
            fields @timestamp, op, elapsed_s
            | filter event = "PROFILE"
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
