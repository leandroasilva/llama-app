import Foundation

/// Builds requests to Hugging Face -- the single place that decides which
/// headers go out, so the API lookup, the metadata HEADs and the downloads
/// can't drift apart.
enum HFRequest {
  /// A request carrying the app's User-Agent, plus the user's token when `url`
  /// is on Hugging Face.
  ///
  /// The header decision (which headers, and that the token only ever goes to
  /// `huggingface.co`) lives in the core (`hfRequest.make`); this only
  /// translates the result. `AppInfo.userAgent` stays Swift-derived -- it comes
  /// from the bundle and OS, not from HF.
  static func make(_ url: URL, token: String?) -> URLRequest {
    var request = URLRequest(url: url)
    var args: [String: Any] = [
      "url": url.absoluteString,
      "userAgent": AppInfo.userAgent,
    ]
    if let token { args["token"] = token }
    guard let out = LlamaCore.callJSON("hfRequest.make", args),
      let headers = out["headers"] as? [[String]]
    else { return request }
    for header in headers { request.setValue(header[1], forHTTPHeaderField: header[0]) }
    return request
  }

  /// The request to perform when a task is redirected. Hugging Face hands the
  /// actual bytes off to a CDN, so the hop must be followed -- but `make` only
  /// ever attaches the token to Hugging Face URLs, and a redirect can point
  /// anywhere. Re-derive the rule for the new URL and drop `Authorization`
  /// whenever the hop leaves Hugging Face, so the token can't be carried off by
  /// a redirect (e.g. HF → `cas-bridge.xethub.hf.co`).
  static func redirectRequest(_ request: URLRequest) -> URLRequest {
    let isHF: Bool = LlamaCore.callValue(
      "hfRequest.isHuggingfaceHost", "value",
      ["url": request.url?.absoluteString ?? ""]) ?? false
    guard !isHF else { return request }
    var stripped = request
    stripped.setValue(nil, forHTTPHeaderField: "Authorization")
    return stripped
  }
}
