use crate::types::{ObservedResponse, Prediction};

/// Evaluate if an observed response matches what was predicted.
#[must_use]
pub fn matches_prediction(
    prediction: &Prediction,
    observed: &ObservedResponse,
    size_tolerance_ratio: f32,
) -> bool {
    if let Some(status) = prediction.status {
        if status != observed.status {
            return false;
        }
    }

    if let Some(content_type) = &prediction.content_type {
        if observed.content_type.as_deref() != Some(content_type.as_str()) {
            return false;
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    if let Some(size) = prediction.approximate_size {
        let tolerance_ratio = size_tolerance_ratio.max(0.0);
        let tolerance = ((size as f32) * tolerance_ratio).round() as usize;
        let min = size.saturating_sub(tolerance);
        let max = size.saturating_add(tolerance);
        if observed.approximate_size < min || observed.approximate_size > max {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_prediction() -> Prediction {
        Prediction {
            status: Some(200),
            content_type: Some("application/json".to_string()),
            approximate_size: Some(100),
            confidence: 0.95,
            samples: 4,
        }
    }

    #[test]
    fn matches_prediction_accepts_exact_match() {
        let observed = ObservedResponse::new(200, Some("application/json"), 100);
        assert!(matches_prediction(&base_prediction(), &observed, 0.0));
    }

    #[test]
    fn matches_prediction_rejects_wrong_status() {
        let observed = ObservedResponse::new(404, Some("application/json"), 100);
        assert!(!matches_prediction(&base_prediction(), &observed, 0.0));
    }

    #[test]
    fn matches_prediction_rejects_wrong_content_type() {
        let observed = ObservedResponse::new(200, Some("text/html"), 100);
        assert!(!matches_prediction(&base_prediction(), &observed, 0.0));
    }

    #[test]
    fn matches_prediction_accepts_size_within_tolerance() {
        let observed = ObservedResponse::new(200, Some("application/json"), 110);
        assert!(matches_prediction(&base_prediction(), &observed, 0.10));
    }

    #[test]
    fn matches_prediction_rejects_size_outside_tolerance() {
        let observed = ObservedResponse::new(200, Some("application/json"), 130);
        assert!(!matches_prediction(&base_prediction(), &observed, 0.10));
    }

    #[test]
    fn matches_prediction_ignores_missing_status_prediction() {
        let mut prediction = base_prediction();
        prediction.status = None;
        let observed = ObservedResponse::new(500, Some("application/json"), 100);
        assert!(matches_prediction(&prediction, &observed, 0.0));
    }

    #[test]
    fn matches_prediction_ignores_missing_content_type_prediction() {
        let mut prediction = base_prediction();
        prediction.content_type = None;
        let observed = ObservedResponse::new(200, Some("text/html"), 100);
        assert!(matches_prediction(&prediction, &observed, 0.0));
    }

    #[test]
    fn matches_prediction_clamps_negative_tolerance() {
        let observed = ObservedResponse::new(200, Some("application/json"), 101);
        assert!(!matches_prediction(&base_prediction(), &observed, -1.0));
    }
}
