
import os
import base64
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def derive_key(self, salt: bytes) -> bytes:
        return PBKDF2(self.password, salt, dkLen=32, count=1000000)

    def encrypt_file(self, input_path: str, output_path: str) -> None:
        salt = get_random_bytes(self.salt_length)
        key = self.derive_key(salt)

        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(key, AES.MODE_CBC, iv)

        with open(input_path, 'rb') as f:
            plaintext = f.read()

        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

        with open(output_path, 'wb') as f:
            f.write(salt + iv + ciphertext)

    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                data = f.read()

            salt = data[:self.salt_length]
            iv = data[self.salt_length:self.salt_length + AES.block_size]
            ciphertext = data[self.salt_length + AES.block_size:]

            key = self.derive_key(salt)
            cipher = AES.new(key, AES.MODE_CBC, iv)

            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

            with open(output_path, 'wb') as f:
                f.write(plaintext)
            return True
        except Exception:
            return False

def main():
    import sys
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input> <output> <password>")
        return

    mode, input_file, output_file, password = sys.argv[1:5]
    encryptor = FileEncryptor(password)

    if mode == 'encrypt':
        encryptor.encrypt_file(input_file, output_file)
        print(f"Encrypted {input_file} -> {output_file}")
    elif mode == 'decrypt':
        success = encryptor.decrypt_file(input_file, output_file)
        if success:
            print(f"Decrypted {input_file} -> {output_file}")
        else:
            print("Decryption failed: wrong password or corrupted file")
    else:
        print("Invalid mode. Use 'encrypt' or 'decrypt'")

if __name__ == "__main__":
    main()