
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def derive_key(self, salt: bytes, iterations: int = 100000) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        return kdf.derive(self.password)

    def encrypt_file(self, input_path: str, output_path: str) -> None:
        salt = os.urandom(self.salt_length)
        key = self.derive_key(salt)
        iv = os.urandom(16)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        with open(input_path, 'rb') as f_in:
            plaintext = f_in.read()

        padding_length = 16 - (len(plaintext) % 16)
        plaintext += bytes([padding_length]) * padding_length

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        with open(output_path, 'wb') as f_out:
            f_out.write(salt + iv + ciphertext)

    def decrypt_file(self, input_path: str, output_path: str) -> None:
        with open(input_path, 'rb') as f_in:
            data = f_in.read()

        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length + 16]
        ciphertext = data[self.salt_length + 16:]

        key = self.derive_key(salt)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        padding_length = plaintext[-1]
        plaintext = plaintext[:-padding_length]

        with open(output_path, 'wb') as f_out:
            f_out.write(plaintext)

def main():
    encryptor = FileEncryptor("secure_password_123")
    
    test_data = b"This is a secret message that needs encryption."
    with open("test_plain.txt", "wb") as f:
        f.write(test_data)

    encryptor.encrypt_file("test_plain.txt", "test_encrypted.bin")
    encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt")

    with open("test_decrypted.txt", "rb") as f:
        result = f.read()
    
    print("Original:", test_data)
    print("Decrypted:", result)
    print("Match:", test_data == result)

    for fname in ["test_plain.txt", "test_encrypted.bin", "test_decrypted.txt"]:
        if os.path.exists(fname):
            os.remove(fname)

if __name__ == "__main__":
    main()
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
            
            while chunk := f_in.read(4096):
                padded_data = padder.update(chunk)
                f_out.write(encryptor.update(padded_data))
            
            final_padded = padder.finalize()
            f_out.write(encryptor.update(final_padded))
            f_out.write(encryptor.finalize())

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
                while chunk := f_in.read(4096):
                    decrypted_data = decryptor.update(chunk)
                    unpadded_data = unpadder.update(decrypted_data)
                    f_out.write(unpadded_data)
                
                final_decrypted = decryptor.finalize()
                final_unpadded = unpadder.update(final_decrypted) + unpadder.finalize()
                f_out.write(final_unpadded)

def generate_secure_password(length: int = 32) -> str:
    return base64.b64encode(os.urandom(length)).decode()[:length]

if __name__ == "__main__":
    test_file = "test_document.txt"
    encrypted_file = "encrypted.dat"
    decrypted_file = "decrypted.txt"
    
    with open(test_file, 'w') as f:
        f.write("Sensitive data that requires protection.")
    
    password = generate_secure_password()
    encryptor = FileEncryptor(password)
    
    encryptor.encrypt_file(test_file, encrypted_file)
    encryptor.decrypt_file(encrypted_file, decrypted_file)
    
    with open(decrypted_file, 'r') as f:
        print(f"Decrypted content: {f.read()}")
    
    os.remove(test_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)