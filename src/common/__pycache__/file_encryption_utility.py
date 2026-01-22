import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import hashlib

def derive_key(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, 32)

def encrypt_file(input_path: str, output_path: str, password: str) -> None:
    salt = os.urandom(16)
    key = derive_key(password, salt)
    iv = os.urandom(16)
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(128).padder()
    
    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        f_out.write(salt + iv)
        
        while chunk := f_in.read(4096):
            padded_data = padder.update(chunk)
            encrypted_chunk = encryptor.update(padded_data)
            f_out.write(encrypted_chunk)
        
        final_padded = padder.finalize()
        final_encrypted = encryptor.update(final_padded) + encryptor.finalize()
        f_out.write(final_encrypted)

def decrypt_file(input_path: str, output_path: str, password: str) -> None:
    with open(input_path, 'rb') as f_in:
        salt = f_in.read(16)
        iv = f_in.read(16)
        key = derive_key(password, salt)
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        unpadder = padding.PKCS7(128).unpadder()
        
        with open(output_path, 'wb') as f_out:
            while chunk := f_in.read(4096):
                decrypted_chunk = decryptor.update(chunk)
                unpadded_data = unpadder.update(decrypted_chunk)
                f_out.write(unpadded_data)
            
            final_decrypted = decryptor.finalize()
            final_unpadded = unpadder.update(final_decrypted) + unpadder.finalize()
            f_out.write(final_unpadded)

def generate_checksum(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def validate_encryption(original_path: str, decrypted_path: str) -> bool:
    return generate_checksum(original_path) == generate_checksum(decrypted_path)