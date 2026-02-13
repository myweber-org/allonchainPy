import os
from cryptography.fernet import Fernet

class FileEncryptor:
    def __init__(self, key_file='secret.key'):
        self.key_file = key_file
        self.key = self.load_or_generate_key()

    def load_or_generate_key(self):
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            return key

    def encrypt_file(self, input_file, output_file=None):
        if output_file is None:
            output_file = input_file + '.encrypted'

        fernet = Fernet(self.key)
        
        with open(input_file, 'rb') as f:
            original_data = f.read()

        encrypted_data = fernet.encrypt(original_data)

        with open(output_file, 'wb') as f:
            f.write(encrypted_data)

        return output_file

    def decrypt_file(self, input_file, output_file=None):
        if not input_file.endswith('.encrypted'):
            raise ValueError("File must have .encrypted extension")

        if output_file is None:
            output_file = input_file.replace('.encrypted', '.decrypted')

        fernet = Fernet(self.key)
        
        with open(input_file, 'rb') as f:
            encrypted_data = f.read()

        try:
            decrypted_data = fernet.decrypt(encrypted_data)
        except:
            raise ValueError("Decryption failed. Invalid key or corrupted file.")

        with open(output_file, 'wb') as f:
            f.write(decrypted_data)

        return output_file

    def encrypt_string(self, text):
        fernet = Fernet(self.key)
        return fernet.encrypt(text.encode()).decode()

    def decrypt_string(self, encrypted_text):
        fernet = Fernet(self.key)
        return fernet.decrypt(encrypted_text.encode()).decode()

def main():
    encryptor = FileEncryptor()
    
    # Example usage
    test_file = 'test_document.txt'
    
    # Create a test file
    with open(test_file, 'w') as f:
        f.write('This is a secret message.')
    
    # Encrypt the file
    encrypted_file = encryptor.encrypt_file(test_file)
    print(f"Encrypted file created: {encrypted_file}")
    
    # Decrypt the file
    decrypted_file = encryptor.decrypt_file(encrypted_file)
    print(f"Decrypted file created: {decrypted_file}")
    
    # Clean up test files
    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)
    
    # String encryption example
    secret_text = "Confidential information"
    encrypted_text = encryptor.encrypt_string(secret_text)
    print(f"Encrypted string: {encrypted_text}")
    
    decrypted_text = encryptor.decrypt_string(encrypted_text)
    print(f"Decrypted string: {decrypted_text}")

if __name__ == "__main__":
    main()
import os
import sys

def xor_cipher(data, key):
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    try:
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        ciphertext = xor_cipher(plaintext, key)
        with open(output_path, 'wb') as f:
            f.write(ciphertext)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)

    operation = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    try:
        key = int(sys.argv[4]) % 256
    except ValueError:
        print("Key must be an integer")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found")
        sys.exit(1)

    if operation == 'encrypt':
        success = encrypt_file(input_file, output_file, key)
    elif operation == 'decrypt':
        success = decrypt_file(input_file, output_file, key)
    else:
        print("Operation must be 'encrypt' or 'decrypt'")
        sys.exit(1)

    if success:
        print(f"Operation '{operation}' completed successfully")
    else:
        print(f"Operation '{operation}' failed")

if __name__ == "__main__":
    main()