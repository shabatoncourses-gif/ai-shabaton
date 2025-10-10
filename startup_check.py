# startup_check.py — בדיקת סביבה לפני ריצה
import os
import sys
import json

def check_env_var(key, required=True):
    """בודק אם משתנה סביבה קיים ומחזיר אזהרה אם חסר"""
    value = os.getenv(key)
    if not value and required:
        print(f"⚠️  Environment variable missing: {key}")
        return False
    return True

def check_dir(path, create=False):
    """בודק אם תיקיה קיימת"""
    if not os.path.exists(path):
        if create:
            os.makedirs(path, exist_ok=True)
            print(f"📁 Created missing directory: {path}")
        else:
            print(f"⚠️  Missing directory: {path}")
            return False
    return True

def check_index(path):
    """בודק אם קיימים קבצי אינדקס בסיסיים"""
    if not os.path.exists(path):
        print(f"⚠️  Index path not found: {path}")
        return
    files = os.listdir(path)
    if not files:
        print("⚠️  Index folder is empty.")
    else:
        print(f"✅ Index directory contains {len(files)} files.")

def check_requirements():
    """בודק התקנות בסיסיות"""
    try:
        import fastapi, chromadb, openai, bs4, requests
        print("✅ Required packages loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to import required packages: {e}")
        sys.exit(1)

def main():
    print("🚀 Running startup environment checks...\n")

    ok = True
    ok &= check_env_var("OPENAI_API_KEY")
    ok &= check_env_var("CHROMA_DB_DIR", required=False)
    ok &= check_env_var("SITEMAP_URL_1", required=False)
    ok &= check_env_var("SITEMAP_URL_2", required=False)

    check_dir("data", create=True)
    check_dir("data/index", create=True)
    check_dir("data/pages", create=True)
    check_index("data/index")
    check_requirements()

    if ok:
        print("\n✅ Environment check completed — all good!\n")
    else:
        print("\n⚠️ Environment incomplete — please review warnings above.\n")

if __name__ == "__main__":
    main()
