use crate::types::{HeaderMap, ObservedResponse, Prediction, RequestContext, SkipPolicy};
use std::collections::BTreeMap;
use url::Url;

/// A predictive model that observes and forecasts HTTP responses.
#[derive(Debug, Clone, Default)]
pub struct ResponsePredictor {
    exact_matches: BTreeMap<String, ResponseStats>,
    families: BTreeMap<String, ResponseStats>,
    hosts: BTreeMap<String, ResponseStats>,
}

impl ResponsePredictor {
    /// Create a new empty predictor model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a real response corresponding to a request.
    pub fn train(&mut self, request: &RequestContext, observed: &ObservedResponse) {
        if let Some(parts) = UrlParts::parse(&request.url, &request.method, &request.headers) {
            self.exact_matches
                .entry(parts.exact_key())
                .or_default()
                .record(observed);
            self.families
                .entry(parts.family_key())
                .or_default()
                .record(observed);
            self.hosts
                .entry(parts.host_key())
                .or_default()
                .record(observed);
        }
    }

    /// Batch train the model with multiple request-response pairs.
    pub fn train_batch<I>(&mut self, data: I)
    where
        I: IntoIterator<Item = (RequestContext, ObservedResponse)>,
    {
        for (request, observed) in data {
            self.train(&request, &observed);
        }
    }

    /// Statistically predict the expected response for a given request shape.
    #[must_use]
    pub fn predict(&self, request: &RequestContext) -> Option<Prediction> {
        let parts = UrlParts::parse(&request.url, &request.method, &request.headers)?;
        self.exact_matches
            .get(&parts.exact_key())
            .or_else(|| self.families.get(&parts.family_key()))
            .or_else(|| self.hosts.get(&parts.host_key()))
            .map(ResponseStats::prediction)
    }

    /// Ask the model whether this request should be skipped according to the policy.
    #[must_use]
    pub fn should_skip(&self, request: &RequestContext, policy: &SkipPolicy) -> bool {
        let Some(parts) = UrlParts::parse(&request.url, &request.method, &request.headers) else {
            return false;
        };

        let stats = self
            .exact_matches
            .get(&parts.exact_key())
            .or_else(|| self.families.get(&parts.family_key()))
            .or_else(|| self.hosts.get(&parts.host_key()));

        let Some(stats) = stats else {
            return false;
        };

        let prediction = stats.prediction();
        prediction.samples >= policy.min_samples
            && prediction.confidence >= policy.min_confidence
            && stats.size_spread_ratio() <= policy.max_size_spread_ratio
    }
}

#[derive(Debug, Clone, Default)]
struct ResponseStats {
    samples: usize,
    statuses: BTreeMap<u16, usize>,
    content_types: BTreeMap<String, usize>,
    size_total: usize,
    size_min: usize,
    size_max: usize,
}

impl ResponseStats {
    fn record(&mut self, observed: &ObservedResponse) {
        self.samples += 1;
        *self.statuses.entry(observed.status).or_insert(0) += 1;
        if let Some(content_type) = &observed.content_type {
            *self.content_types.entry(content_type.clone()).or_insert(0) += 1;
        }
        self.size_total += observed.approximate_size;
        self.size_min = if self.samples == 1 {
            observed.approximate_size
        } else {
            self.size_min.min(observed.approximate_size)
        };
        self.size_max = self.size_max.max(observed.approximate_size);
    }

    #[allow(clippy::cast_precision_loss)]
    fn prediction(&self) -> Prediction {
        let (status, status_count) = dominant_key(&self.statuses);
        let (content_type, content_type_count) = dominant_key(&self.content_types);
        let confidence = if self.samples == 0 {
            0.0
        } else {
            let status_conf = status_count as f32 / self.samples as f32;
            let content_conf = if self.content_types.is_empty() {
                status_conf
            } else {
                content_type_count as f32 / self.samples as f32
            };
            (status_conf * 0.7) + (content_conf * 0.3)
        };

        Prediction {
            status: status.copied(),
            content_type: content_type.cloned(),
            approximate_size: (self.samples > 0).then(|| self.size_total / self.samples),
            confidence,
            samples: self.samples,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn size_spread_ratio(&self) -> f32 {
        if self.samples == 0 || self.size_total == 0 {
            return 1.0;
        }
        let avg = (self.size_total / self.samples).max(1) as f32;
        (self.size_max.saturating_sub(self.size_min) as f32) / avg
    }
}

#[derive(Debug, Clone)]
struct UrlParts {
    method: String,
    host: String,
    exact_path: String,
    family_path: String,
    query_shape: String,
    header_signature: String,
}

impl UrlParts {
    fn parse(url: &str, method: &str, headers: &HeaderMap) -> Option<Self> {
        let parsed = Url::parse(url).ok()?;
        let host = parsed.host_str()?.to_string();
        let exact_path = normalized_exact_path(&parsed);
        let family_path = normalized_family_path(&parsed);
        let query_shape = normalized_query_shape(&parsed);
        let header_signature = normalized_header_signature(headers);

        Some(Self {
            method: method.to_uppercase(),
            host,
            exact_path,
            family_path,
            query_shape,
            header_signature,
        })
    }

    fn exact_key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.method, self.host, self.exact_path, self.query_shape, self.header_signature
        )
    }

    fn family_key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.method, self.host, self.family_path, self.query_shape, self.header_signature
        )
    }

    fn host_key(&self) -> String {
        format!("{}|{}|{}", self.method, self.host, self.header_signature)
    }
}

fn dominant_key<K: Ord>(map: &BTreeMap<K, usize>) -> (Option<&K>, usize) {
    map.iter()
        .max_by_key(|(_, count)| **count)
        .map_or((None, 0), |(key, count)| (Some(key), *count))
}

fn normalized_exact_path(url: &Url) -> String {
    let path = url.path();
    if path.is_empty() {
        "/".to_string()
    } else {
        path.to_string()
    }
}

fn normalized_family_path(url: &Url) -> String {
    let segments = url
        .path_segments()
        .map(|segments| segments.map(classify_segment).collect::<Vec<_>>())
        .unwrap_or_default();

    if segments.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", segments.join("/"))
    }
}

fn normalized_query_shape(url: &Url) -> String {
    let mut keys = url
        .query_pairs()
        .map(|(key, _)| key.into_owned())
        .collect::<Vec<_>>();
    keys.sort();
    keys.join("&")
}

fn normalized_header_signature(headers: &HeaderMap) -> String {
    let mut parts = Vec::new();
    for key in [
        "accept",
        "authorization",
        "content-type",
        "x-requested-with",
    ] {
        if let Some(value) = headers
            .iter()
            .find(|(header, _)| header.eq_ignore_ascii_case(key))
            .map(|(_, value)| value)
        {
            parts.push(format!("{}={}", key, value.to_ascii_lowercase()));
        }
    }
    parts.join("|")
}

fn classify_segment(segment: &str) -> String {
    if segment.is_empty() {
        return String::new();
    }
    if segment.chars().all(|character| character.is_ascii_digit()) {
        return "{int}".to_string();
    }
    if looks_like_uuid(segment) {
        return "{uuid}".to_string();
    }
    if segment.len() >= 8
        && segment
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return "{hex}".to_string();
    }
    segment.to_ascii_lowercase()
}

fn looks_like_uuid(segment: &str) -> bool {
    let parts = segment.split('-').collect::<Vec<_>>();
    matches!(parts.as_slice(), [p1, p2, p3, p4, p5]
        if p1.len() == 8
            && p2.len() == 4
            && p3.len() == 4
            && p4.len() == 4
            && p5.len() == 12
            && parts
                .iter()
                .all(|part| part.chars().all(|character| character.is_ascii_hexdigit())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(url: &str) -> RequestContext {
        RequestContext::new(url, "GET").with_header("Accept", "application/json")
    }

    fn observed(status: u16, content_type: Option<&str>, size: usize) -> ObservedResponse {
        ObservedResponse::new(status, content_type, size)
    }

    #[test]
    fn predictor_new_starts_empty() {
        let predictor = ResponsePredictor::new();
        assert!(predictor.exact_matches.is_empty());
        assert!(predictor.families.is_empty());
        assert!(predictor.hosts.is_empty());
    }

    #[test]
    fn train_records_exact_match_entry() {
        let mut predictor = ResponsePredictor::new();
        predictor.train(
            &request("https://example.com/users/1"),
            &observed(200, Some("application/json"), 64),
        );
        assert_eq!(predictor.exact_matches.len(), 1);
    }

    #[test]
    fn train_ignores_invalid_urls() {
        let mut predictor = ResponsePredictor::new();
        predictor.train(
            &RequestContext::new("not-a-url", "GET"),
            &observed(200, Some("application/json"), 64),
        );
        assert!(predictor.exact_matches.is_empty());
    }

    #[test]
    fn train_batch_records_multiple_samples() {
        let mut predictor = ResponsePredictor::new();
        predictor.train_batch(vec![
            (
                request("https://example.com/users/1"),
                observed(200, Some("application/json"), 64),
            ),
            (
                request("https://example.com/users/1"),
                observed(200, Some("application/json"), 66),
            ),
        ]);
        assert_eq!(
            predictor
                .predict(&request("https://example.com/users/1"))
                .unwrap()
                .samples,
            2
        );
    }

    #[test]
    fn predict_returns_none_for_empty_training_data() {
        let predictor = ResponsePredictor::new();
        assert_eq!(
            predictor.predict(&request("https://example.com/users/1")),
            None
        );
    }

    #[test]
    fn predict_prefers_exact_match() {
        let mut predictor = ResponsePredictor::new();
        predictor.train(
            &request("https://example.com/users/1"),
            &observed(201, Some("application/json"), 70),
        );
        predictor.train(
            &request("https://example.com/users/2"),
            &observed(404, Some("text/plain"), 10),
        );
        let prediction = predictor
            .predict(&request("https://example.com/users/1"))
            .unwrap();
        assert_eq!(prediction.status, Some(201));
    }

    #[test]
    fn predict_falls_back_to_family_match() {
        let mut predictor = ResponsePredictor::new();
        predictor.train(
            &request("https://example.com/users/1"),
            &observed(200, Some("application/json"), 64),
        );
        let prediction = predictor
            .predict(&request("https://example.com/users/2"))
            .unwrap();
        assert_eq!(prediction.status, Some(200));
    }

    #[test]
    fn predict_falls_back_to_host_match() {
        let mut predictor = ResponsePredictor::new();
        predictor.train(
            &request("https://example.com/users/1"),
            &observed(202, Some("application/json"), 90),
        );
        let other = RequestContext::new("https://example.com/admin", "GET")
            .with_header("Accept", "application/json");
        let prediction = predictor.predict(&other).unwrap();
        assert_eq!(prediction.status, Some(202));
    }

    #[test]
    fn predict_tracks_content_type() {
        let mut predictor = ResponsePredictor::new();
        predictor.train(
            &request("https://example.com/api"),
            &observed(200, Some("application/json"), 64),
        );
        let prediction = predictor
            .predict(&request("https://example.com/api"))
            .unwrap();
        assert_eq!(prediction.content_type.as_deref(), Some("application/json"));
    }

    #[test]
    fn predict_tracks_average_size() {
        let mut predictor = ResponsePredictor::new();
        predictor.train(
            &request("https://example.com/api"),
            &observed(200, Some("application/json"), 50),
        );
        predictor.train(
            &request("https://example.com/api"),
            &observed(200, Some("application/json"), 70),
        );
        let prediction = predictor
            .predict(&request("https://example.com/api"))
            .unwrap();
        assert_eq!(prediction.approximate_size, Some(60));
    }

    #[test]
    fn prediction_confidence_reflects_majority_vote() {
        let mut stats = ResponseStats::default();
        stats.record(&observed(200, Some("application/json"), 10));
        stats.record(&observed(200, Some("application/json"), 12));
        stats.record(&observed(500, Some("text/plain"), 9));
        let prediction = stats.prediction();
        assert!(prediction.confidence > 0.5);
        assert_eq!(prediction.status, Some(200));
    }

    #[test]
    fn should_skip_requires_sufficient_samples_and_confidence() {
        let mut predictor = ResponsePredictor::new();
        predictor.train(
            &request("https://example.com/api"),
            &observed(200, Some("application/json"), 50),
        );
        predictor.train(
            &request("https://example.com/api"),
            &observed(200, Some("application/json"), 52),
        );
        predictor.train(
            &request("https://example.com/api"),
            &observed(200, Some("application/json"), 54),
        );
        assert!(predictor.should_skip(&request("https://example.com/api"), &SkipPolicy::default()));
    }

    #[test]
    fn should_skip_rejects_high_size_spread() {
        let mut predictor = ResponsePredictor::new();
        predictor.train(
            &request("https://example.com/api"),
            &observed(200, Some("application/json"), 10),
        );
        predictor.train(
            &request("https://example.com/api"),
            &observed(200, Some("application/json"), 100),
        );
        predictor.train(
            &request("https://example.com/api"),
            &observed(200, Some("application/json"), 200),
        );
        assert!(!predictor.should_skip(&request("https://example.com/api"), &SkipPolicy::default()));
    }

    #[test]
    fn normalized_family_path_rewrites_numeric_segments() {
        let url = Url::parse("https://example.com/users/123/orders/456").unwrap();
        assert_eq!(normalized_family_path(&url), "/users/{int}/orders/{int}");
    }

    #[test]
    fn normalized_family_path_rewrites_uuid_segments() {
        let url =
            Url::parse("https://example.com/users/550e8400-e29b-41d4-a716-446655440000").unwrap();
        assert_eq!(normalized_family_path(&url), "/users/{uuid}");
    }

    #[test]
    fn normalized_family_path_rewrites_hex_segments() {
        let url = Url::parse("https://example.com/build/abcdef12").unwrap();
        assert_eq!(normalized_family_path(&url), "/build/{hex}");
    }

    #[test]
    fn normalized_query_shape_sorts_keys() {
        let url = Url::parse("https://example.com/api?b=2&a=1").unwrap();
        assert_eq!(normalized_query_shape(&url), "a&b");
    }

    #[test]
    fn normalized_header_signature_extracts_selected_headers() {
        let headers = HeaderMap::from([
            ("Accept".to_string(), "APPLICATION/JSON".to_string()),
            ("X-Ignored".to_string(), "value".to_string()),
        ]);
        assert_eq!(
            normalized_header_signature(&headers),
            "accept=application/json"
        );
    }

    #[test]
    fn classify_segment_preserves_words() {
        assert_eq!(classify_segment("Users"), "users");
    }

    #[test]
    fn looks_like_uuid_recognizes_uuid_shape() {
        assert!(looks_like_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!looks_like_uuid("not-a-uuid"));
    }
}
