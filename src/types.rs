use std::collections::BTreeMap;

/// Mapping of string headers to string values.
pub type HeaderMap = BTreeMap<String, String>;

/// A minimal request context used for predictions.
///
/// This struct captures the essential information about an HTTP request
/// that is needed for training the prediction model and making predictions.
///
/// # Examples
///
/// ```
/// use respredict::RequestContext;
///
/// let request = RequestContext::new("https://api.example.com/users/1", "GET")
///     .with_header("Accept", "application/json")
///     .with_header("Authorization", "Bearer token123");
///
/// assert_eq!(request.url, "https://api.example.com/users/1");
/// assert_eq!(request.method, "GET");
/// assert_eq!(request.headers.get("Accept"), Some(&"application/json".to_string()));
/// ```
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
    ///
    /// The HTTP method will be normalized to uppercase.
    ///
    /// # Arguments
    ///
    /// * `url` - The target URL for the request.
    /// * `method` - The HTTP method (e.g., "GET", "POST").
    ///
    /// # Returns
    ///
    /// A new `RequestContext` with empty headers.
    ///
    /// # Examples
    ///
    /// ```
    /// use respredict::RequestContext;
    ///
    /// let request = RequestContext::new("https://example.com/api", "post");
    /// assert_eq!(request.method, "POST"); // Normalized to uppercase
    /// ```
    #[must_use]
    pub fn new(url: impl Into<String>, method: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            method: method.into().to_uppercase(),
            headers: HeaderMap::new(),
        }
    }

    /// Add a header to the request context fluently.
    ///
    /// This method returns `self` to allow chaining multiple headers.
    ///
    /// # Arguments
    ///
    /// * `key` - The header name.
    /// * `value` - The header value.
    ///
    /// # Returns
    ///
    /// The modified `RequestContext` for method chaining.
    ///
    /// # Examples
    ///
    /// ```
    /// use respredict::RequestContext;
    ///
    /// let request = RequestContext::new("https://example.com", "GET")
    ///     .with_header("Accept", "application/json")
    ///     .with_header("Content-Type", "application/json");
    ///
    /// assert_eq!(request.headers.len(), 2);
    /// ```
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

/// The observed ground-truth response for a request.
///
/// This struct represents the actual response received from a server,
/// used to train and evaluate the prediction model.
///
/// # Examples
///
/// ```
/// use respredict::ObservedResponse;
///
/// let response = ObservedResponse::new(200, Some("application/json"), 1024);
///
/// assert_eq!(response.status, 200);
/// assert_eq!(response.content_type, Some("application/json".to_string()));
/// assert_eq!(response.approximate_size, 1024);
/// ```
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
    ///
    /// # Arguments
    ///
    /// * `status` - The HTTP status code (e.g., 200, 404).
    /// * `content_type` - Optional MIME type of the response body.
    /// * `approximate_size` - Approximate size of the response body in bytes.
    ///
    /// # Returns
    ///
    /// A new `ObservedResponse` with the specified fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use respredict::ObservedResponse;
    ///
    /// // With content type
    /// let response = ObservedResponse::new(200, Some("text/html"), 512);
    ///
    /// // Without content type (e.g., 204 No Content)
    /// let empty_response = ObservedResponse::new(204, None, 0);
    /// ```
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
///
/// This struct contains the predicted characteristics of an HTTP response,
/// along with confidence metrics based on historical observations.
///
/// # Examples
///
/// ```
/// use respredict::Prediction;
///
/// let prediction = Prediction {
///     status: Some(200),
///     content_type: Some("application/json".to_string()),
///     approximate_size: Some(256),
///     confidence: 0.95,
///     samples: 10,
/// };
///
/// assert!(prediction.confidence > 0.9);
/// assert_eq!(prediction.samples, 10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// The predicted status code.
    pub status: Option<u16>,
    /// The predicted content type.
    pub content_type: Option<String>,
    /// The predicted approximate size.
    pub approximate_size: Option<usize>,
    /// How confident the model evaluates this prediction (0.0 to 1.0).
    pub confidence: f32,
    /// How many samples were observed for this shape.
    pub samples: usize,
}

/// A policy governing when to skip a request based on confidence.
///
/// The `SkipPolicy` defines the thresholds that must be met before
/// the predictor will recommend skipping a request.
///
/// # Default Values
///
/// * `min_confidence`: 0.85 (85% confidence required)
/// * `min_samples`: 3 (at least 3 observations needed)
/// * `max_size_spread_ratio`: 0.20 (size variance must be within 20%)
///
/// # Examples
///
/// ```
/// use respredict::SkipPolicy;
///
/// // Use default policy
/// let policy = SkipPolicy::default();
///
/// // Create a custom policy
/// let strict_policy = SkipPolicy {
///     min_confidence: 0.95,
///     min_samples: 5,
///     max_size_spread_ratio: 0.10,
/// };
/// ```
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
    /// Returns the default skip policy.
    ///
    /// # Default Values
    ///
    /// * `min_confidence`: 0.85
    /// * `min_samples`: 3
    /// * `max_size_spread_ratio`: 0.20
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
