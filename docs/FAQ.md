# Ruvrics – Frequently Asked Questions (FAQ)

This document captures common issues, pitfalls, and clarifications for users of **ruvrics**. It will evolve over time as new questions arise.

---

## ❓ I installed `ruvrics` from TestPyPI, but it shows an older version

### **What I see**
I installed `ruvrics` from TestPyPI, but `pip show ruvrics` shows an older version than expected.

```bash
pip install --index-url https://test.pypi.org/simple/ ruvrics
pip show ruvrics
```

---

### **Why this happens**
This is a common TestPyPI + pip behavior and is **not an error** in `ruvrics`.

Possible reasons:

1. **pip cache**  
   `pip` may reuse a cached wheel or metadata from a previous install.

2. **TestPyPI metadata delay**  
   TestPyPI can lag when updating the "latest version" pointer, so pip may resolve an older release.

3. **Dependencies are not hosted on TestPyPI**  
   TestPyPI does not host packages like NumPy or Torch. Without a fallback to the main PyPI index, resolution may behave inconsistently.

4. **An older version is already installed**  
   pip does not always upgrade an existing package automatically.

---

### **How to fix it (recommended)**

Install with an explicit version, disable cache, and allow dependency resolution from the main PyPI index:

```bash
pip uninstall -y ruvrics
pip cache purge
pip install \
  --no-cache-dir \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  ruvrics==0.1.3
```

Verify the installed version:

```bash
pip show ruvrics
```

---

### **Helpful checks**

See which versions are available on TestPyPI:

```bash
pip index versions ruvrics --index-url https://test.pypi.org/simple/
```

Check whether pip has cached an older wheel:

```bash
pip cache list | grep ruvrics
```

---

### **Best practice when using TestPyPI**

- Use an **explicit version** when installing from TestPyPI
- Include `--extra-index-url https://pypi.org/simple` for dependencies
- Prefer a **fresh virtual environment** when testing
- Use `--no-cache-dir` when validating new installs

---

More FAQs will be added here as the project evolves.

