"I have completed a thorough read of all 10 files. Here is the full bug audit report. I'll organize it by file.

---

# ð Bug Audit Report â Final Project v1.0

---

## ð `config.py`

### Bug 1 â Wrong `DATA_DIR` Path (Critical)
- **Line:** 27
- **Severity:** Critical
- **Description:** `DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")` goes **one level above** the project root (i.e., `../data`). Since `BASE_DIR` is already the project root (where `config.py` lives), this creates the `data/` folder *outside* the project directory. All downstream paths (`PDF_DIR`, `VECTOR_DB_DIR`, etc.) will be stored in the parent folder.
- **Fix:**
```python
# Wrong:
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
# Correct:
DATA_DIR = os.path.join(BASE_DIR, "data")
```

---

## ð `app.py`

### Bug 2 â Idle Monitor Re-Spawns on Every Streamlit Rerun (Medium)
- **Lines:** 46â71
- **Severity:** Medium
- **Description:** The `GLOBAL_LAST_ACTIVE` guard (`if 'GLOBAL_LAST_ACTIVE' not in globals()`) is module-level code. Every time Streamlit reruns the script in the **same Python process**, the `globals()` check on line 46 will always be `True` (since it was defined), but line 50 still **unconditionally resets** `GLOBAL_LAST_ACTIVE = time.time()`. This means the idle timer is reset on every user interaction, which is actually desired â but the thread-name guard on lines 63â71 is unreliable because Streamlit may run app.py in different threads, and `threading.enumerate()` is process-global so there is a small race window on first startup. More importantly, `global GLOBAL_LAST_ACTIVE` declared inside an `if` block at module level (line 47) is a no-op and confusing; the real reset on line 50 works but the `if` guard on line 46 is misleading dead code.
- **Fix:** Remove lines 46â48 entirely (the `if 'GLOBAL_LAST_ACTIVE' not in globals()` guard is unnecessary and misleading). Keep only line 50. Add a threading lock around the thread-check loop to prevent a race 
<truncated 24346 bytes>