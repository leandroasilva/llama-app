import Foundation
import Security

/// Minimal Keychain wrapper for storing sensitive values (tokens, secrets).
///
/// Uses the app's default access group so items survive app updates but are
/// removed on uninstall. No iCloud sync -- these are machine-local secrets.
enum KeychainStore {
  private static let service = "app.llama.Llama"

  /// Stores a string value under the given key. Overwrites any existing value.
  @discardableResult
  static func set(_ value: String, forKey key: String) -> Bool {
    guard let data = value.data(using: .utf8) else { return false }

    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: key,
    ]

    let attributes: [String: Any] = [
      kSecValueData as String: data,
      kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
    ]

    let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
    if status == errSecItemNotFound {
      var addQuery = query
      addQuery.merge(attributes) { _, new in new }
      return SecItemAdd(addQuery as CFDictionary, nil) == errSecSuccess
    }
    return status == errSecSuccess
  }

  /// Retrieves the string value stored under the given key, or nil if absent.
  static func get(forKey key: String) -> String? {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: key,
      kSecReturnData as String: true,
      kSecMatchLimit as String: kSecMatchLimitOne,
    ]

    var result: AnyObject?
    let status = SecItemCopyMatching(query as CFDictionary, &result)
    guard status == errSecSuccess, let data = result as? Data else { return nil }
    return String(data: data, encoding: .utf8)
  }

  /// Removes the value stored under the given key.
  @discardableResult
  static func delete(forKey key: String) -> Bool {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service,
      kSecAttrAccount as String: key,
    ]
    let status = SecItemDelete(query as CFDictionary)
    return status == errSecSuccess || status == errSecItemNotFound
  }
}
