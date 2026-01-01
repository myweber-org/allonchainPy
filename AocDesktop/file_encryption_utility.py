
import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password):
        self.password = password.encode('utf-8')
        self.salt = get_random_bytes(16)
        self.key = PBKDF2(self.password, self.salt, dkLen=32, count=1000000)

    def encrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        iv = get_random_bytes(16)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)

        with open(input_path, 'rb') as f:
            plaintext = f.read()

        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

        if output_path is None:
            output_path = input_path + '.enc'

        with open(output_path, 'wb') as f:
            f.write(self.salt + iv + ciphertext)

        return output_path

    def decrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'rb') as f:
            data = f.read()

        salt = data[:16]
        iv = data[16:32]
        ciphertext = data[32:]

        key = PBKDF2(self.password, salt, dkLen=32, count=1000000)
        cipher = AES.new(key, AES.MODE_CBC, iv)

        try:
            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        except ValueError as e:
            raise ValueError("Decryption failed. Incorrect password or corrupted file.") from e

        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'

        with open(output_path, 'wb') as f:
            f.write(plaintext)

        return output_path

    def calculate_hash(self, file_path, algorithm='sha256'):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)

        return hash_func.hexdigest()

def example_usage():
    password = "secure_password_123"
    encryptor = FileEncryptor(password)

    test_file = "test_document.txt"
    with open(test_file, 'w') as f:
        f.write("This is a secret document containing sensitive information.")

    try:
        encrypted_file = encryptor.encrypt_file(test_file)
        print(f"Encrypted file created: {encrypted_file}")

        original_hash = encryptor.calculate_hash(test_file)
        print(f"Original file hash: {original_hash}")

        decrypted_file = encryptor.decrypt_file(encrypted_file)
        print(f"Decrypted file created: {decrypted_file}")

        decrypted_hash = encryptor.calculate_hash(decrypted_file)
        print(f"Decrypted file hash: {decrypted_hash}")

        if original_hash == decrypted_hash:
            print("Encryption/decryption successful: Hashes match")
        else:
            print("Error: Hashes do not match")

    finally:
        for file in [test_file, encrypted_file, decrypted_file]:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    example_usage()