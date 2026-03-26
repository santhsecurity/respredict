use std::collections::BTreeMap;

/// Mapping of string headers to string values.
pub type HeaderMap = BTreeMap<String, String>;

/// A minimal request context used for predictions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestContext {
    /// The target URL.
    pub url: String,
    /// The HTTP method.
    pub method: String,
    /// The request headers.
    pub headers: HeaderMap,
}

impl RequestContext {
    /// Create a new request context.
    #[must_use]
    pub fn new(url: impl Into<String>, method: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            method: method.into().to_uppercase(),
            headers: HeaderMap::new(),
        }
    }

    /// Add a header to the request context fluently.
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

/// The observed ground-truth response for a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedResponse {
    /// The HTTP status code.
    pub status: u16,
    /// The inferred content type, if any.
    pub content_type: Option<String>,
    /// The approximate size in bytes.
    pub approximate_size: usize,
}

impl ObservedResponse {
    /// Create a new observed response.
    #[must_use]
    pub fn new(status: u16, content_type: Option<&str>, approximate_size: usize) -> Self {
        Self {
            status,
            content_type: content_type.map(std::string::ToString::to_string),
            approximate_size,
        }
    }
}

/// A statistical prediction for a response.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// The predicted status code.
    pub status: Option<u16>,
    /// The predicted content type.
    pub content_type: Option<String>,
    /// The predicted approximate size.
    pub approximate_size: Option<usize>,
    /// How confident the model evaluates this prediction.
    pub confidence: f32,
    /// How many samples were observed for this shape.
    pub samples: usize,
}

/// A policy governing when to skip a request based on confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct SkipPolicy {
    /// Minimum confidence threshold (0.0 to 1.0).
    pub min_confidence: f32,
    /// Minimum sample count.
    pub min_samples: usize,
    /// Maximum allowed variance across size measurements.
    pub max_size_spread_ratio: f32,
}

impl Default for SkipPolicy {
    fn default() -> Self {
        Self {
            min_confidence: 0.85,
            min_samples: 3,
            max_size_spread_ratio: 0.20,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_context_new_uppercases_method() {
        let request = RequestContext::new("https://example.com", "post");
        assert_eq!(request.method, "POST");
    }

    #[test]
    fn request_context_with_header_adds_value() {
        let request = RequestContext::new("https://example.com", "GET")
            .with_header("Accept", "application/json");
        assert_eq!(
            request.headers.get("Accept"),
            Some(&"application/json".to_string())
        );
    }

    #[test]
    fn observed_response_new_sets_fields() {
        let observed = ObservedResponse::new(200, Some("text/html"), 128);
        assert_eq!(observed.status, 200);
        assert_eq!(observed.content_type.as_deref(), Some("text/html"));
        assert_eq!(observed.approximate_size, 128);
    }

    #[test]
    fn observed_response_new_handles_missing_content_type() {
        let observed = ObservedResponse::new(204, None, 0);
        assert_eq!(observed.content_type, None);
    }

    #[test]
    fn skip_policy_default_values_are_stable() {
        let policy = SkipPolicy::default();
        assert_eq!(policy.min_confidence, 0.85);
        assert_eq!(policy.min_samples, 3);
        assert_eq!(policy.max_size_spread_ratio, 0.20);
    }

    #[test]
    fn prediction_supports_partial_equality() {
        let prediction = Prediction {
            status: Some(200),
            content_type: Some("application/json".to_string()),
            approximate_size: Some(64),
            confidence: 0.9,
            samples: 4,
        };
        assert_eq!(prediction.clone(), prediction);
    }

    #[test]
    fn header_map_alias_behaves_like_btreemap() {
        let mut headers = HeaderMap::new();
        headers.insert("accept".to_string(), "text/plain".to_string());
        assert_eq!(headers.get("accept"), Some(&"text/plain".to_string()));
    }

    #[test]
    fn request_context_starts_with_empty_headers() {
        let request = RequestContext::new("https://example.com", "GET");
        assert!(request.headers.is_empty());
    }
}
