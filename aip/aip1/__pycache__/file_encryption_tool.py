import os
import sys

def xor_cipher(data: bytes, key: str) -> bytes:
    """Apply XOR cipher to data with given key."""
    key_bytes = key.encode()
    return bytes([data[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(data))])

def encrypt_file(input_path: str, output_path: str, key: str):
    """Encrypt a file using XOR cipher."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        encrypted_data = xor_cipher(data, key)
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        print(f"Encryption successful. Output saved to: {output_path}")
    except Exception as e:
        print(f"Error during encryption: {e}")

def decrypt_file(input_path: str, output_path: str, key: str):
    """Decrypt a file using XOR cipher (same as encryption)."""
    encrypt_file(input_path, output_path, key)
    print(f"Decryption successful. Output saved to: {output_path}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    key = sys.argv[4]
    
    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)
    
    if mode == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif mode == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Invalid mode. Use 'encrypt' or 'decrypt'.")

if __name__ == "__main__":
    main()