
import os
import sys

def xor_cipher(data, key):
    """Encrypt or decrypt data using XOR cipher."""
    return bytes([b ^ key for b in data])

def process_file(input_path, output_path, key):
    """Encrypt or decrypt a file."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        processed_data = xor_cipher(data, key)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"File processed successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key>")
        print("Key must be an integer between 0 and 255")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        key = int(sys.argv[3])
        if not 0 <= key <= 255:
            raise ValueError
    except ValueError:
        print("Error: Key must be an integer between 0 and 255")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    if process_file(input_file, output_file, key):
        print("Operation completed successfully")
    else:
        print("Operation failed")

if __name__ == "__main__":
    main()
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def encrypt(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def decrypt(self, data: bytes) -> bytes:
        return self.encrypt(data)

def process_file(input_path: str, output_path: str, key: str, mode: str):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    try:
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        if mode == 'encrypt':
            processed_data = cipher.encrypt(file_data)
            action = "Encrypted"
        elif mode == 'decrypt':
            processed_data = cipher.decrypt(file_data)
            action = "Decrypted"
        else:
            print("Error: Mode must be 'encrypt' or 'decrypt'.")
            sys.exit(1)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"{action} file saved to: {output_path}")
        print(f"Original size: {len(file_data)} bytes")
        print(f"Processed size: {len(processed_data)} bytes")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key> <mode>")
        print("Mode: 'encrypt' or 'decrypt'")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]
    mode = sys.argv[4].lower()
    
    process_file(input_file, output_file, key, mode)

if __name__ == "__main__":
    main()import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')

    def _xor_operation(self, data: bytes) -> bytes:
        key_bytes = self.key
        key_length = len(key_bytes)
        return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])

    def encrypt_file(self, input_path: str, output_path: str):
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            ciphertext = self._xor_operation(plaintext)
            with open(output_path, 'wb') as f:
                f.write(ciphertext)
            print(f"File encrypted successfully: {output_path}")
        except Exception as e:
            print(f"Encryption failed: {e}")

    def decrypt_file(self, input_path: str, output_path: str):
        try:
            with open(input_path, 'rb') as f:
                ciphertext = f.read()
            plaintext = self._xor_operation(ciphertext)
            with open(output_path, 'wb') as f:
                f.write(plaintext)
            print(f"File decrypted successfully: {output_path}")
        except Exception as e:
            print(f"Decryption failed: {e}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <key> <input_file> <output_file>")
        sys.exit(1)

    operation = sys.argv[1].lower()
    key = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)

    cipher = XORCipher(key)

    if operation == 'encrypt':
        cipher.encrypt_file(input_file, output_file)
    elif operation == 'decrypt':
        cipher.decrypt_file(input_file, output_file)
    else:
        print("Invalid operation. Use 'encrypt' or 'decrypt'.")

if __name__ == "__main__":
    main()