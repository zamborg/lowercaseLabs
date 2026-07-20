import CloudKit
import CryptoKit
import Foundation

enum AppleICloudIdentity {
    static let containerID = "iCloud.com.lowercaseLabs.theVoid"

    static func requireAvailableICloudAccount() async throws {
        let status = await accountStatus()
        guard status == .available else {
            throw AppleICloudIdentityError.iCloudUnavailable(statusDescription(status))
        }
    }

    static func accountStatus() async -> CKAccountStatus {
        await withCheckedContinuation { continuation in
            CKContainer(identifier: containerID).accountStatus { status, _ in
                continuation.resume(returning: status)
            }
        }
    }

    static func localUserID(for appleUserID: String) -> String {
        "apple-\(hash(appleUserID).prefix(32))"
    }

    static func anonymousHandle(for appleUserID: String) -> String {
        "void-\(hash(appleUserID).prefix(8))"
    }

    private static func hash(_ value: String) -> String {
        SHA256.hash(data: Data(value.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func statusDescription(_ status: CKAccountStatus) -> String {
        switch status {
        case .available:
            return "available"
        case .couldNotDetermine:
            return "could not determine"
        case .noAccount:
            return "no iCloud account"
        case .restricted:
            return "restricted"
        case .temporarilyUnavailable:
            return "temporarily unavailable"
        @unknown default:
            return "unknown"
        }
    }
}

private enum AppleICloudIdentityError: LocalizedError {
    case iCloudUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .iCloudUnavailable(let status):
            return "Sign in to iCloud to use theVoid on this device. iCloud status: \(status)."
        }
    }
}
