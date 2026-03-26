use respredict::{ObservedResponse, RequestContext, ResponsePredictor};

fn main() {
    let mut predictor = ResponsePredictor::new();
    let request = RequestContext::new("https://example.com/missing", "GET")
        .with_header("Accept", "text/html");

    predictor.train(
        &request,
        &ObservedResponse::new(404, Some("text/html"), 512),
    );
    predictor.train(
        &request,
        &ObservedResponse::new(404, Some("text/html"), 520),
    );
    predictor.train(
        &request,
        &ObservedResponse::new(404, Some("text/html"), 500),
    );

    let prediction = predictor.predict(&request).unwrap();
    println!("{prediction:?}");
}
