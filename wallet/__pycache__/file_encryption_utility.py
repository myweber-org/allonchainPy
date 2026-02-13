import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

class FileEncryptor:
    def __init__(self, password: str, salt: bytes = None):
        self.password = password.encode()
        self.salt = salt or os.urandom(16)
        self.key = self._derive_key()

    def _derive_key(self):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(self.password)

    def encrypt_file(self, input_path: str, output_path: str):
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()

        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            f_out.write(self.salt)
            f_out.write(iv)
            
            while True:
                chunk = f_in.read(4096)
                if not chunk:
                    break
                padded_data = padder.update(chunk)
                encrypted_chunk = encryptor.update(padded_data)
                f_out.write(encrypted_chunk)
            
            final_padded = padder.finalize()
            final_encrypted = encryptor.update(final_padded) + encryptor.finalize()
            f_out.write(final_encrypted)

    def decrypt_file(self, input_path: str, output_path: str):
        with open(input_path, 'rb') as f_in:
            salt = f_in.read(16)
            iv = f_in.read(16)
            
            if salt != self.salt:
                self.salt = salt
                self.key = self._derive_key()
            
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            unpadder = padding.PKCS7(128).unpadder()
            
            with open(output_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(4096)
                    if not chunk:
                        break
                    decrypted_chunk = decryptor.update(chunk)
                    unpadded_data = unpadder.update(decrypted_chunk)
                    f_out.write(unpadded_data)
                
                final_decrypted = decryptor.finalize()
                final_unpadded = unpadder.update(final_decrypted) + unpadder.finalize()
                f_out.write(final_unpadded)

def generate_random_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode()

if __name__ == "__main__":
    encryptor = FileEncryptor("secure_password_123")
    
    test_data = b"This is a secret message that needs encryption."
    with open("test_original.txt", "wb") as f:
        f.write(test_data)
    
    encryptor.encrypt_file("test_original.txt", "test_encrypted.bin")
    encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt")
    
    with open("test_decrypted.txt", "rb") as f:
        print(f.read())
    
    os.remove("test_original.txt")
    os.remove("test_encrypted.bin")
    os.remove("test_decrypted.txt")import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(self.password)

    def encrypt_file(self, input_path: str, output_path: str) -> None:
        salt = os.urandom(self.salt_length)
        key = self._derive_key(salt)

        iv = os.urandom(16)
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            f_out.write(salt)
            f_out.write(iv)

            while True:
                chunk = f_in.read(1024 * 64)
                if not chunk:
                    break
                if len(chunk) % 16 != 0:
                    chunk += b' ' * (16 - len(chunk) % 16)
                encrypted_chunk = encryptor.update(chunk)
                f_out.write(encrypted_chunk)

            f_out.write(encryptor.finalize())

    def decrypt_file(self, input_path: str, output_path: str) -> None:
        with open(input_path, 'rb') as f_in:
            salt = f_in.read(self.salt_length)
            iv = f_in.read(16)
            key = self._derive_key(salt)

            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            with open(output_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(1024 * 64)
                    if not chunk:
                        break
                    decrypted_chunk = decryptor.update(chunk)
                    f_out.write(decrypted_chunk)

                f_out.write(decryptor.finalize().rstrip(b' '))

def main():
    encryptor = FileEncryptor("secure_password_123")
    
    test_data = b"This is a secret message that needs encryption."
    with open("test_original.txt", "wb") as f:
        f.write(test_data)

    encryptor.encrypt_file("test_original.txt", "test_encrypted.bin")
    encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt")

    with open("test_decrypted.txt", "rb") as f:
        decrypted = f.read()
    
    print("Original equals decrypted:", test_data == decrypted)
    
    os.remove("test_original.txt")
    os.remove("test_encrypted.bin")
    os.remove("test_decrypted.txt")

if __name__ == "__main__":
    main()