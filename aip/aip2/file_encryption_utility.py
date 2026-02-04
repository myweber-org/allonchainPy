
import os
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
    
    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            
            salt = os.urandom(self.salt_length)
            key = self._derive_key(salt)
            iv = os.urandom(16)
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            pad_length = 16 - (len(plaintext) % 16)
            padded_data = plaintext + bytes([pad_length] * pad_length)
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            encrypted_data = salt + iv + ciphertext
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            return True
        except Exception:
            return False
    
    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()
            
            salt = encrypted_data[:self.salt_length]
            iv = encrypted_data[self.salt_length:self.salt_length + 16]
            ciphertext = encrypted_data[self.salt_length + 16:]
            
            key = self._derive_key(salt)
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            pad_length = padded_plaintext[-1]
            plaintext = padded_plaintext[:-pad_length]
            
            with open(output_path, 'wb') as f:
                f.write(plaintext)
            
            return True
        except Exception:
            return False
    
    def encrypt_string(self, plaintext: str) -> str:
        salt = os.urandom(self.salt_length)
        key = self._derive_key(salt)
        iv = os.urandom(16)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        data = plaintext.encode()
        pad_length = 16 - (len(data) % 16)
        padded_data = data + bytes([pad_length] * pad_length)
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        combined = salt + iv + ciphertext
        return base64.b64encode(combined).decode()
    
    def decrypt_string(self, encrypted_string: str) -> str:
        combined = base64.b64decode(encrypted_string)
        salt = combined[:self.salt_length]
        iv = combined[self.salt_length:self.salt_length + 16]
        ciphertext = combined[self.salt_length + 16:]
        
        key = self._derive_key(salt)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        pad_length = padded_plaintext[-1]
        plaintext = padded_plaintext[:-pad_length]
        
        return plaintext.decode()