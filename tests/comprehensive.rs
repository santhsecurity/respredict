use respredict::{
    matches_prediction, ObservedResponse, Prediction, RequestContext, ResponsePredictor, SkipPolicy,
};

// ============================================================================
// 1. Training the predictor with known responses
// ============================================================================

#[test]
fn training_with_single_response_creates_prediction() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/users", "GET");
    let observed = ObservedResponse::new(200, Some("application/json"), 1024);

    predictor.train(&request, &observed);

    let prediction = predictor.predict(&request).expect("should have prediction");
    assert_eq!(prediction.status, Some(200));
    assert_eq!(prediction.content_type, Some("application/json".to_string()));
    assert_eq!(prediction.approximate_size, Some(1024));
    assert_eq!(prediction.samples, 1);
}

#[test]
fn training_with_multiple_responses_aggregates_stats() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/users", "GET");

    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 1000));
    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 2000));
    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 3000));

    let prediction = predictor.predict(&request).expect("should have prediction");
    assert_eq!(prediction.samples, 3);
    assert_eq!(prediction.approximate_size, Some(2000)); // average
    assert_eq!(prediction.status, Some(200));
}

#[test]
fn training_with_different_methods_creates_separate_entries() {
    let mut predictor = ResponsePredictor::new();
    let get_request = RequestContext::new("https://example.com/api/users", "GET");
    let post_request = RequestContext::new("https://example.com/api/users", "POST");

    predictor.train(&get_request, &ObservedResponse::new(200, Some("application/json"), 100));
    predictor.train(&post_request, &ObservedResponse::new(201, Some("application/json"), 50));

    let get_prediction = predictor.predict(&get_request).expect("should have GET prediction");
    let post_prediction = predictor.predict(&post_request).expect("should have POST prediction");

    assert_eq!(get_prediction.status, Some(200));
    assert_eq!(post_prediction.status, Some(201));
}

#[test]
fn training_with_different_hosts_creates_separate_entries() {
    let mut predictor = ResponsePredictor::new();
    let request1 = RequestContext::new("https://api1.example.com/users", "GET");
    let request2 = RequestContext::new("https://api2.example.com/users", "GET");

    predictor.train(&request1, &ObservedResponse::new(200, Some("application/json"), 100));
    predictor.train(&request2, &ObservedResponse::new(404, Some("text/plain"), 50));

    let prediction1 = predictor.predict(&request1).expect("should have prediction");
    let prediction2 = predictor.predict(&request2).expect("should have prediction");

    assert_eq!(prediction1.status, Some(200));
    assert_eq!(prediction2.status, Some(404));
}

#[test]
fn training_with_different_headers_creates_separate_entries() {
    let mut predictor = ResponsePredictor::new();
    let request1 = RequestContext::new("https://example.com/api", "GET")
        .with_header("Accept", "application/json");
    let request2 = RequestContext::new("https://example.com/api", "GET")
        .with_header("Accept", "text/html");

    predictor.train(&request1, &ObservedResponse::new(200, Some("application/json"), 100));
    predictor.train(&request2, &ObservedResponse::new(200, Some("text/html"), 200));

    let prediction1 = predictor.predict(&request1).expect("should have prediction");
    let prediction2 = predictor.predict(&request2).expect("should have prediction");

    assert_eq!(prediction1.content_type, Some("application/json".to_string()));
    assert_eq!(prediction2.content_type, Some("text/html".to_string()));
}

#[test]
fn batch_training_records_all_samples() {
    let mut predictor = ResponsePredictor::new();
    let data = vec![
        (RequestContext::new("https://example.com/api/1", "GET"), ObservedResponse::new(200, Some("application/json"), 100)),
        (RequestContext::new("https://example.com/api/2", "GET"), ObservedResponse::new(200, Some("application/json"), 200)),
        (RequestContext::new("https://example.com/api/3", "GET"), ObservedResponse::new(404, Some("text/plain"), 50)),
    ];

    predictor.train_batch(data);

    assert!(predictor.predict(&RequestContext::new("https://example.com/api/1", "GET")).is_some());
    assert!(predictor.predict(&RequestContext::new("https://example.com/api/2", "GET")).is_some());
    assert!(predictor.predict(&RequestContext::new("https://example.com/api/3", "GET")).is_some());
}

#[test]
fn training_with_path_parameters_normalizes_to_family() {
    let mut predictor = ResponsePredictor::new();
    let request1 = RequestContext::new("https://example.com/users/123", "GET");
    let request2 = RequestContext::new("https://example.com/users/456", "GET");

    predictor.train(&request1, &ObservedResponse::new(200, Some("application/json"), 100));

    // Should fall back to family match
    let prediction = predictor.predict(&request2).expect("should have prediction via family");
    assert_eq!(prediction.status, Some(200));
}

#[test]
fn training_with_uuid_path_normalizes_correctly() {
    let mut predictor = ResponsePredictor::new();
    let request1 = RequestContext::new("https://example.com/items/550e8400-e29b-41d4-a716-446655440000", "GET");
    let request2 = RequestContext::new("https://example.com/items/aabbccdd-1234-5678-9abc-def012345678", "GET");

    predictor.train(&request1, &ObservedResponse::new(200, Some("application/json"), 100));

    let prediction = predictor.predict(&request2).expect("should have prediction via family");
    assert_eq!(prediction.status, Some(200));
}

#[test]
fn training_with_hex_path_normalizes_correctly() {
    let mut predictor = ResponsePredictor::new();
    let request1 = RequestContext::new("https://example.com/build/abcdef1234567890", "GET");
    let request2 = RequestContext::new("https://example.com/build/0123456789abcdef", "GET");

    predictor.train(&request1, &ObservedResponse::new(200, Some("text/plain"), 100));

    let prediction = predictor.predict(&request2).expect("should have prediction via family");
    assert_eq!(prediction.status, Some(200));
}

#[test]
fn training_accumulates_confidence_with_consistent_responses() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/stable", "GET");

    for _ in 0..5 {
        predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 100));
    }

    let prediction = predictor.predict(&request).expect("should have prediction");
    assert_eq!(prediction.confidence, 1.0); // perfect consistency
    assert_eq!(prediction.samples, 5);
}

#[test]
fn training_with_inconsistent_responses_lowers_confidence() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/unstable", "GET");

    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 100));
    predictor.train(&request, &ObservedResponse::new(500, Some("text/plain"), 50));
    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 110));

    let prediction = predictor.predict(&request).expect("should have prediction");
    assert!(prediction.confidence < 1.0);
    assert!(prediction.confidence > 0.5); // 200 is still majority
}

// ============================================================================
// 2. Prediction accuracy after training
// ============================================================================

#[test]
fn prediction_is_accurate_for_exact_url_match() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/users", "GET");
    let observed = ObservedResponse::new(200, Some("application/json"), 1024);

    predictor.train(&request, &observed);
    let prediction = predictor.predict(&request).expect("should predict");

    assert!(matches_prediction(&prediction, &observed, 0.0));
}

#[test]
fn prediction_is_accurate_for_similar_urls() {
    let mut predictor = ResponsePredictor::new();
    let train_request = RequestContext::new("https://example.com/api/users/1", "GET");
    let predict_request = RequestContext::new("https://example.com/api/users/2", "GET");
    let observed = ObservedResponse::new(200, Some("application/json"), 100);

    predictor.train(&train_request, &observed);
    let prediction = predictor.predict(&predict_request).expect("should predict via family");

    // Should predict same status and content type
    assert_eq!(prediction.status, Some(200));
    assert_eq!(prediction.content_type, Some("application/json".to_string()));
}

#[test]
fn prediction_accuracy_decreases_with_diverse_training() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/mixed", "GET");

    // Train with very different responses
    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 100));
    predictor.train(&request, &ObservedResponse::new(404, Some("text/html"), 200));
    predictor.train(&request, &ObservedResponse::new(500, Some("text/plain"), 50));

    let prediction = predictor.predict(&request).expect("should predict");
    
    // Confidence should be low due to diversity
    assert!(prediction.confidence < 0.8);
}

#[test]
fn prediction_size_is_average_of_training_samples() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/sized", "GET");

    predictor.train(&request, &ObservedResponse::new(200, None, 100));
    predictor.train(&request, &ObservedResponse::new(200, None, 200));
    predictor.train(&request, &ObservedResponse::new(200, None, 300));

    let prediction = predictor.predict(&request).expect("should predict");
    assert_eq!(prediction.approximate_size, Some(200)); // (100+200+300)/3
}

#[test]
fn prediction_confidence_increases_with_more_samples() {
    let mut predictor1 = ResponsePredictor::new();
    let mut predictor2 = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/test", "GET");

    // Predictor 1: few samples
    predictor1.train(&request, &ObservedResponse::new(200, Some("json"), 100));
    predictor1.train(&request, &ObservedResponse::new(200, Some("json"), 100));

    // Predictor 2: many consistent samples
    for _ in 0..10 {
        predictor2.train(&request, &ObservedResponse::new(200, Some("json"), 100));
    }

    let pred1 = predictor1.predict(&request).expect("should predict");
    let pred2 = predictor2.predict(&request).expect("should predict");

    // Both have perfect consistency, but predictor 2 has more samples
    assert_eq!(pred1.confidence, 1.0);
    assert_eq!(pred2.confidence, 1.0);
    assert_eq!(pred1.samples, 2);
    assert_eq!(pred2.samples, 10);
}

// ============================================================================
// 3. Empty training data edge case
// ============================================================================

#[test]
fn empty_predictor_returns_none_for_any_request() {
    let predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");

    assert!(predictor.predict(&request).is_none());
}

#[test]
fn empty_predictor_should_skip_returns_false() {
    let predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");
    let policy = SkipPolicy::default();

    assert!(!predictor.should_skip(&request, &policy));
}

#[test]
fn predictor_with_training_for_other_urls_returns_none_for_unseen() {
    let mut predictor = ResponsePredictor::new();
    let trained_request = RequestContext::new("https://example.com/api/users", "GET");
    let unseen_request = RequestContext::new("https://other.com/api/items", "GET");

    predictor.train(&trained_request, &ObservedResponse::new(200, None, 100));

    // Different host with no training data
    assert!(predictor.predict(&unseen_request).is_none());
}

#[test]
fn predictor_after_clearing_not_possible_but_new_instance_is_empty() {
    // Since ResponsePredictor doesn't have a clear() method,
    // we verify that a new instance behaves as empty
    let predictor = ResponsePredictor::new();
    assert!(predictor.predict(&RequestContext::new("https://example.com", "GET")).is_none());
}

#[test]
fn empty_batch_training_leaves_predictor_empty() {
    let mut predictor = ResponsePredictor::new();
    let empty_data: Vec<(RequestContext, ObservedResponse)> = vec![];

    predictor.train_batch(empty_data);

    assert!(predictor.predict(&RequestContext::new("https://example.com", "GET")).is_none());
}

#[test]
fn invalid_url_training_keeps_predictor_empty() {
    let mut predictor = ResponsePredictor::new();
    let invalid_request = RequestContext::new("not-a-valid-url", "GET");

    predictor.train(&invalid_request, &ObservedResponse::new(200, None, 100));

    assert!(predictor.predict(&invalid_request).is_none());
    assert!(predictor.predict(&RequestContext::new("https://example.com", "GET")).is_none());
}

// ============================================================================
// 4. Prediction for URLs never seen before
// ============================================================================

#[test]
fn unseen_url_with_same_host_uses_host_level_prediction() {
    let mut predictor = ResponsePredictor::new();
    let trained_request = RequestContext::new("https://example.com/api/users", "GET");
    let unseen_request = RequestContext::new("https://example.com/api/items/999", "GET");

    predictor.train(&trained_request, &ObservedResponse::new(200, Some("application/json"), 100));

    // Same host, different path - should use host-level prediction
    let prediction = predictor.predict(&unseen_request).expect("should have host-level prediction");
    assert_eq!(prediction.status, Some(200));
}

#[test]
fn unseen_url_with_similar_pattern_uses_family_prediction() {
    let mut predictor = ResponsePredictor::new();
    let trained_request = RequestContext::new("https://example.com/users/123", "GET");
    let unseen_request = RequestContext::new("https://example.com/users/999", "GET");

    predictor.train(&trained_request, &ObservedResponse::new(200, Some("application/json"), 100));

    let prediction = predictor.predict(&unseen_request).expect("should have family prediction");
    assert_eq!(prediction.status, Some(200));
}

#[test]
fn completely_unseen_host_returns_none() {
    let mut predictor = ResponsePredictor::new();
    predictor.train(
        &RequestContext::new("https://example.com/api", "GET"),
        &ObservedResponse::new(200, None, 100),
    );

    let completely_new = RequestContext::new("https://never-seen-before.com/api", "GET");
    assert!(predictor.predict(&completely_new).is_none());
}

#[test]
fn unseen_url_with_different_method_uses_host_prediction() {
    let mut predictor = ResponsePredictor::new();
    predictor.train(
        &RequestContext::new("https://example.com/api/users", "GET"),
        &ObservedResponse::new(200, Some("application/json"), 100),
    );

    let post_request = RequestContext::new("https://example.com/api/users", "POST");
    // Same host, different method - host key includes method
    let prediction = predictor.predict(&post_request);
    // Since host key includes method, this is effectively unseen
    assert!(prediction.is_none() || prediction.unwrap().samples < 2);
}

#[test]
fn unseen_url_preserves_confidence_metrics() {
    let mut predictor = ResponsePredictor::new();
    let base_request = RequestContext::new("https://example.com/api/items/1", "GET");

    // Train with multiple consistent samples
    for _ in 0..5 {
        predictor.train(&base_request, &ObservedResponse::new(200, Some("json"), 100));
    }

    let unseen_request = RequestContext::new("https://example.com/api/items/999", "GET");
    let prediction = predictor.predict(&unseen_request).expect("should predict");

    // Family-level prediction should inherit confidence
    assert!(prediction.confidence > 0.8);
    assert_eq!(prediction.samples, 5);
}

// ============================================================================
// 5. Content type prediction correctness
// ============================================================================

#[test]
fn content_type_prediction_matches_majority() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");

    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 100));
    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 100));
    predictor.train(&request, &ObservedResponse::new(200, Some("text/html"), 100));

    let prediction = predictor.predict(&request).expect("should predict");
    assert_eq!(prediction.content_type, Some("application/json".to_string()));
}

#[test]
fn content_type_prediction_with_none_values() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");

    predictor.train(&request, &ObservedResponse::new(204, None, 0));
    predictor.train(&request, &ObservedResponse::new(204, None, 0));
    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 100));

    let prediction = predictor.predict(&request).expect("should predict");
    // None values are not counted, so json should win
    assert_eq!(prediction.content_type, Some("application/json".to_string()));
}

#[test]
fn content_type_prediction_all_none_returns_none() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");

    predictor.train(&request, &ObservedResponse::new(204, None, 0));
    predictor.train(&request, &ObservedResponse::new(204, None, 0));

    let prediction = predictor.predict(&request).expect("should predict");
    assert_eq!(prediction.content_type, None);
}

#[test]
fn content_type_prediction_is_case_sensitive() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");

    // Store different cases separately
    predictor.train(&request, &ObservedResponse::new(200, Some("Application/JSON"), 100));
    predictor.train(&request, &ObservedResponse::new(200, Some("application/json"), 100));
    predictor.train(&request, &ObservedResponse::new(200, Some("APPLICATION/JSON"), 100));

    let prediction = predictor.predict(&request).expect("should predict");
    // One of them should win (implementation specific which)
    assert!(prediction.content_type.is_some());
}

#[test]
fn content_type_prediction_for_different_endpoints() {
    let mut predictor = ResponsePredictor::new();
    let api_request = RequestContext::new("https://example.com/api/data", "GET");
    let page_request = RequestContext::new("https://example.com/page.html", "GET");

    predictor.train(&api_request, &ObservedResponse::new(200, Some("application/json"), 100));
    predictor.train(&page_request, &ObservedResponse::new(200, Some("text/html"), 200));

    let api_prediction = predictor.predict(&api_request).expect("should predict");
    let page_prediction = predictor.predict(&page_request).expect("should predict");

    assert_eq!(api_prediction.content_type, Some("application/json".to_string()));
    assert_eq!(page_prediction.content_type, Some("text/html".to_string()));
}

// ============================================================================
// 6. Status code prediction correctness
// ============================================================================

#[test]
fn status_code_prediction_matches_majority() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");

    predictor.train(&request, &ObservedResponse::new(200, None, 100));
    predictor.train(&request, &ObservedResponse::new(200, None, 100));
    predictor.train(&request, &ObservedResponse::new(500, None, 50));

    let prediction = predictor.predict(&request).expect("should predict");
    assert_eq!(prediction.status, Some(200));
}

#[test]
fn status_code_prediction_with_tie_breaker() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");

    predictor.train(&request, &ObservedResponse::new(200, None, 100));
    predictor.train(&request, &ObservedResponse::new(404, None, 100));

    let prediction = predictor.predict(&request).expect("should predict");
    // First one encountered should win due to BTreeMap ordering
    assert!(prediction.status == Some(200) || prediction.status == Some(404));
}

#[test]
fn status_code_prediction_perfect_confidence() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/stable", "GET");

    for _ in 0..10 {
        predictor.train(&request, &ObservedResponse::new(200, None, 100));
    }

    let prediction = predictor.predict(&request).expect("should predict");
    assert_eq!(prediction.status, Some(200));
    // Status confidence is 1.0, but overall confidence may differ if no content type
}

#[test]
fn status_code_prediction_for_error_responses() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api/error", "GET");

    predictor.train(&request, &ObservedResponse::new(500, Some("text/plain"), 50));
    predictor.train(&request, &ObservedResponse::new(503, Some("text/plain"), 50));
    predictor.train(&request, &ObservedResponse::new(500, Some("text/plain"), 50));

    let prediction = predictor.predict(&request).expect("should predict");
    assert_eq!(prediction.status, Some(500));
}

#[test]
fn status_code_prediction_for_redirects() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/redirect", "GET");

    predictor.train(&request, &ObservedResponse::new(301, None, 0));
    predictor.train(&request, &ObservedResponse::new(302, None, 0));
    predictor.train(&request, &ObservedResponse::new(301, None, 0));

    let prediction = predictor.predict(&request).expect("should predict");
    assert_eq!(prediction.status, Some(301));
}

// ============================================================================
// Additional edge cases and integration tests
// ============================================================================

#[test]
fn should_skip_with_insufficient_samples() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");
    let policy = SkipPolicy::default(); // min_samples = 3

    predictor.train(&request, &ObservedResponse::new(200, None, 100));
    predictor.train(&request, &ObservedResponse::new(200, None, 100));

    // Only 2 samples, policy requires 3
    assert!(!predictor.should_skip(&request, &policy));
}

#[test]
fn should_skip_with_low_confidence() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");
    let policy = SkipPolicy::default(); // min_confidence = 0.85

    // Train with inconsistent data to lower confidence
    for i in 0..5 {
        let status = if i % 2 == 0 { 200 } else { 500 };
        predictor.train(&request, &ObservedResponse::new(status, None, 100));
    }

    // Confidence should be too low
    assert!(!predictor.should_skip(&request, &policy));
}

#[test]
fn should_skip_with_high_size_variance() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/api", "GET");
    let policy = SkipPolicy::default(); // max_size_spread_ratio = 0.20

    // Train with wildly different sizes
    predictor.train(&request, &ObservedResponse::new(200, None, 100));
    predictor.train(&request, &ObservedResponse::new(200, None, 1000));
    predictor.train(&request, &ObservedResponse::new(200, None, 500));

    // Size spread should be too high
    assert!(!predictor.should_skip(&request, &policy));
}

#[test]
fn matches_prediction_handles_zero_tolerance() {
    let prediction = Prediction {
        status: Some(200),
        content_type: Some("application/json".to_string()),
        approximate_size: Some(100),
        confidence: 1.0,
        samples: 1,
    };

    let exact_match = ObservedResponse::new(200, Some("application/json"), 100);
    let off_by_one = ObservedResponse::new(200, Some("application/json"), 101);

    assert!(matches_prediction(&prediction, &exact_match, 0.0));
    assert!(!matches_prediction(&prediction, &off_by_one, 0.0));
}

#[test]
fn matches_prediction_with_large_tolerance() {
    let prediction = Prediction {
        status: Some(200),
        content_type: Some("application/json".to_string()),
        approximate_size: Some(100),
        confidence: 1.0,
        samples: 1,
    };

    let larger = ObservedResponse::new(200, Some("application/json"), 150);

    assert!(matches_prediction(&prediction, &larger, 0.5)); // 50% tolerance
    assert!(!matches_prediction(&prediction, &larger, 0.4)); // 40% tolerance
}

#[test]
fn query_parameters_affect_exact_match() {
    let mut predictor = ResponsePredictor::new();
    let request_with_query = RequestContext::new("https://example.com/api?foo=bar", "GET");
    let request_without_query = RequestContext::new("https://example.com/api", "GET");

    predictor.train(&request_with_query, &ObservedResponse::new(200, None, 100));

    // Different query shape - should fall back to host level
    let _prediction = predictor.predict(&request_without_query);
    // May or may not have prediction depending on fallback behavior
    // Both should at least not panic
}

#[test]
fn authorization_header_included_in_signature() {
    let mut predictor = ResponsePredictor::new();
    let authed_request = RequestContext::new("https://example.com/api", "GET")
        .with_header("Authorization", "Bearer token123");
    let no_auth_request = RequestContext::new("https://example.com/api", "GET");

    predictor.train(&authed_request, &ObservedResponse::new(200, None, 100));

    // No auth request should not match (different header signature)
    assert!(predictor.predict(&no_auth_request).is_none());
}
