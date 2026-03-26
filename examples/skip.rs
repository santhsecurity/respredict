use respredict::{ObservedResponse, RequestContext, ResponsePredictor, SkipPolicy};

fn main() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/assets/9999", "GET")
        .with_header("Accept", "text/html");

    for size in [500, 510, 505] {
        predictor.train(
            &request,
            &ObservedResponse::new(404, Some("text/html"), size),
        );
    }

    let should_skip = predictor.should_skip(&request, &SkipPolicy::default());
    println!("skip={should_skip}");
}
