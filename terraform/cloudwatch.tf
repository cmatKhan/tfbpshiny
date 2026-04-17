locals {
  log_group  = "/tfbpshiny/production"
  log_stream = "shinyapp"
  namespace  = "TFBPShiny"
}

# ---------------------------------------------------------------------------
# Metric filters — extract numeric values from pipe-delimited log records
# ---------------------------------------------------------------------------

# Extracts elapsed_s from every PROFILE record.
# Pattern captures the second numeric field after "PROFILE | <ts> | ".
resource "aws_cloudwatch_log_metric_filter" "profile_latency" {
  name           = "tfbpshiny-profile-latency"
  log_group_name = local.log_group
  pattern        = "[marker=PROFILE, ts, elapsed_s, op, module, dataset, context, sid]"

  metric_transformation {
    name          = "ProfileLatencySeconds"
    namespace     = local.namespace
    value         = "$elapsed_s"
    default_value = 0
    unit          = "Seconds"
  }
}

# Counts SESSION START records — one per new user connection.
resource "aws_cloudwatch_log_metric_filter" "session_start" {
  name           = "tfbpshiny-session-start"
  log_group_name = local.log_group
  pattern        = "[marker=SESSION, ts, event=START, sid]"

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

# Fires when avg vdb query latency exceeds 30 s over a 15-minute window.
# Threshold is intentionally high while the app is known to be slow —
# lower it toward 5–10 s once query performance improves.
resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "tfbpshiny-high-latency"
  alarm_description   = "Avg PROFILE latency > 30 s over 15 min — investigate vdb query bottlenecks."
  namespace           = local.namespace
  metric_name         = "ProfileLatencySeconds"
  statistic           = "Average"
  period              = 900  # 15 minutes
  evaluation_periods  = 1
  threshold           = 30
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
  period              = 300  # 5 minutes
  evaluation_periods  = 1
  threshold           = 20
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
}

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

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
          logGroupNames = [local.log_group]
          queryString   ="fields @timestamp, @message | filter @message like /^PROFILE/ | parse @message 'PROFILE | * | * | * | * | * | * | *' as ts, elapsed_s, op, module, dataset, context, sid | stats avg(elapsed_s) as avg_s, max(elapsed_s) as max_s, count() as n by op, module, context | sort avg_s desc"
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title         = "Concurrent sessions started (5-min bins)"
          region        = var.aws_region
          view          = "timeSeries"
          logGroupNames = [local.log_group]
          queryString   ="fields @timestamp, @message | filter @message like /^SESSION/ | filter @message like /START/ | parse @message 'SESSION | * | * | *' as ts, event, sid | stats count(sid) as sessions_started by bin(5m) | sort @timestamp asc"
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
          view          = "table"
          logGroupNames = [local.log_group]
          queryString   ="fields @timestamp, @message | filter @message like /^SESSION/ | filter @message like /START/ | parse @message 'SESSION | * | * | *' as ts, event, sid | stats count_distinct(sid) as unique_visitors by datefloor(@timestamp, 1d) | sort datefloor_timestamp asc"
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
          logGroupNames = [local.log_group]
          queryString   ="fields @timestamp, @message | filter @message like /^PROFILE/ | parse @message 'PROFILE | * | * | * | * | * | * | *' as ts, elapsed_s, op, module, dataset, context, sid | stats avg(elapsed_s) as avg_s by bin(5m), op | sort @timestamp asc"
        }
      },
      {
        type   = "alarm"
        x      = 0
        y      = 18
        width  = 12
        height = 3
        properties = {
          title  = "High latency alarm"
          alarms = [aws_cloudwatch_metric_alarm.high_latency.arn]
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
