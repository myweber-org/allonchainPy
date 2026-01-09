import os
import sys

def xor_cipher(data, key):
    """XOR cipher for encryption and decryption."""
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    """Encrypt a file using XOR cipher."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        encrypted_data = xor_cipher(data, key)
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        print(f"File encrypted successfully: {output_path}")
    except Exception as e:
        print(f"Error during encryption: {e}")

def decrypt_file(input_path, output_path, key):
    """Decrypt a file using XOR cipher."""
    encrypt_file(input_path, output_path, key)  # XOR is symmetric

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    try:
        key = int(sys.argv[4]) % 256
    except ValueError:
        print("Key must be an integer.")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if mode == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif mode == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Mode must be 'encrypt' or 'decrypt'.")

if __name__ == "__main__":
    main()