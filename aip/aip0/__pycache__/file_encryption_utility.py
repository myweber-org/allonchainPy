
import os
import base64
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16, iterations: int = 100000):
        self.password = password.encode()
        self.salt_length = salt_length
        self.iterations = iterations

    def derive_key(self, salt: bytes) -> bytes:
        return PBKDF2(self.password, salt, dkLen=32, count=self.iterations)

    def encrypt_file(self, input_path: str, output_path: str) -> None:
        salt = get_random_bytes(self.salt_length)
        key = self.derive_key(salt)
        
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
        
        with open(output_path, 'wb') as f:
            f.write(salt + iv + ciphertext)

    def decrypt_file(self, input_path: str, output_path: str) -> None:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length + 16]
        ciphertext = data[self.salt_length + 16:]
        
        key = self.derive_key(salt)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)

    def encrypt_string(self, plaintext: str) -> str:
        salt = get_random_bytes(self.salt_length)
        key = self.derive_key(salt)
        
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        
        ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
        combined = salt + iv + ciphertext
        
        return base64.b64encode(combined).decode()

    def decrypt_string(self, encrypted_data: str) -> str:
        data = base64.b64decode(encrypted_data.encode())
        
        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length + 16]
        ciphertext = data[self.salt_length + 16:]
        
        key = self.derive_key(salt)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return plaintext.decode()

def example_usage():
    encryptor = FileEncryptor("secure_password_123")
    
    # Encrypt a string
    encrypted = encryptor.encrypt_string("Sensitive information")
    print(f"Encrypted: {encrypted}")
    
    # Decrypt the string
    decrypted = encryptor.decrypt_string(encrypted)
    print(f"Decrypted: {decrypted}")
    
    # File encryption example (commented out for safety)
    # encryptor.encrypt_file("plain.txt", "encrypted.bin")
    # encryptor.decrypt_file("encrypted.bin", "decrypted.txt")

if __name__ == "__main__":
    example_usage()