#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTF-8 fix for Windows console output
"""

import sys
import io
import os

def setup_utf8_console():
    """Setup UTF-8 encoding for Windows console"""
    if sys.platform == 'win32':
        # Set console code page to UTF-8
        os.system('chcp 65001 > nul')
        
        # Wrap stdout and stderr with UTF-8 encoding
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, 
            encoding='utf-8', 
            errors='replace',
            line_buffering=True
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding='utf-8',
            errors='replace', 
            line_buffering=True
        )

def safe_print(msg, file=None):
    """Print with safe encoding fallback"""
    if file is None:
        file = sys.stdout
    
    try:
        print(msg, file=file, flush=True)
    except UnicodeEncodeError:
        # Replace Turkish characters with ASCII equivalents
        safe_msg = msg
        replacements = {
            'ş': 's', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ç': 'c', 'ü': 'u',
            'Ş': 'S', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ç': 'C', 'Ü': 'U',
            'â': 'a', 'î': 'i', '–': '-', '—': '-', '▶': '>>', '■': '--',
            'ä±': 'i', 'Ã§': 'c', 'ÅŸ': 's', 'Ä°': 'I', 'Ã¶': 'o',
            'Ã¼': 'u', 'ÄŸ': 'g'
        }
        for tr_char, ascii_char in replacements.items():
            safe_msg = safe_msg.replace(tr_char, ascii_char)
        
        print(safe_msg, file=file, flush=True)
    except Exception as e:
        # Final fallback
        try:
            ascii_msg = msg.encode('ascii', 'ignore').decode('ascii')
            print(ascii_msg, file=file, flush=True)
        except:
            pass

# Turkish text mappings for common log messages
TURKISH_TO_ASCII = {
    'Giriş doğrulama': 'Giris dogrulama',
    'Değişken sınıflaması': 'Degisken siniflamasi',
    'değer politikası': 'deger politikasi',
    'bölmesi': 'bolmesi',
    'yalnız': 'yalniz',
    'vektörize': 'vektorize',
    'Gürültü': 'Gurultu',
    'değerlendirme': 'degerlendirme',
    'seçimi': 'secimi',
    'tabloları': 'tablolari',
    'başlıyor': 'basliyor',
    'hazır': 'hazir',
    'değişken': 'degisken',
    'sonrası': 'sonrasi',
    'seçti': 'secti',
    'kaldı': 'kaldi'
}

def fix_turkish_text(text):
    """Replace Turkish characters with ASCII equivalents"""
    result = text
    for turkish, ascii_text in TURKISH_TO_ASCII.items():
        result = result.replace(turkish, ascii_text)
    
    # Individual character replacements
    chars = {
        'ş': 's', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ç': 'c', 'ü': 'u',
        'Ş': 'S', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ç': 'C', 'Ü': 'U'
    }
    for tr_char, ascii_char in chars.items():
        result = result.replace(tr_char, ascii_char)
    
    return result