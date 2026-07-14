# `try`, `except`, `finally` in Python

This is a very common Senior Python interview question because it tests your understanding of **exception handling**, **resource cleanup**, and **production reliability**.

The key difference:

| Keyword   | Purpose                     |
| --------- | --------------------------- |
| `try`     | Code that may fail          |
| `except`  | Handle the error            |
| `finally` | Always execute cleanup code |

---

# Basic Structure

```python
try:
    # risky code

except Exception:
    # handle error

finally:
    # cleanup code
```

Execution flow:

```text
          try
           |
     -------------
     |           |
  Success     Exception
     |           |
     |        except
     |           |
     -------------
           |
        finally
```

---

# 1. `except`

## Purpose

`except` handles exceptions.

Example:

```python
id="except_basic"
try:
    x = 10 / 0

except ZeroDivisionError:
    print("Cannot divide by zero")
```

Output:

```
Cannot divide by zero
```

Without `except`:

```python
id="no_except"
x = 10 / 0
```

Python crashes:

```
ZeroDivisionError
```

---

# Multiple `except` Blocks

Example:

```python
id="multiple_except"
try:

    value = int(input("Enter number: "))

    result = 10 / value


except ValueError:

    print("Invalid number")


except ZeroDivisionError:

    print("Cannot divide by zero")
```

Different errors have different handling.

---

# Catching Generic Exception

```python
id="generic_except"
try:
    process()

except Exception as e:
    print(e)
```

This catches almost all normal exceptions.

Production systems usually log:

```python
id="logging_exception"
except Exception as e:
    logger.error(
        "Failed",
        exc_info=True
    )
```

---

# 2. `finally`

## Purpose

`finally` executes **always**, whether an exception happens or not.

Example:

```python
id="finally_basic"
try:

    x = 10 / 2

except Exception:
    print("Error")

finally:
    print("Cleanup")
```

Output:

```
Cleanup
```

---

Now with error:

```python
id="finally_error"
try:

    x = 10 / 0

except Exception:
    print("Error")

finally:
    print("Cleanup")
```

Output:

```
Error
Cleanup
```

Notice:

`finally` still runs.

---

# Why Do We Need `finally`?

Because some resources must always be released.

Examples:

* Database connections
* Files
* Network sockets
* Locks
* GPU memory
* API sessions

---

# Example: File Handling

Without finally:

```python
id="file_bad"
file = open("data.txt")

data = file.read()

file.close()
```

Problem:

If `read()` fails:

```text
file.close()
```

never executes.

The file remains open.

---

Using finally:

```python
id="file_finally"
file = open("data.txt")

try:

    data = file.read()


except Exception as e:

    print(e)


finally:

    file.close()
```

Now:

Success:

```
read
close
```

Failure:

```
error
close
```

---

# `finally` with return

Very common interview trap.

Example:

```python
id="return_finally"
def test():

    try:
        return 10

    finally:
        print("finally")


print(test())
```

Output:

```
finally
10
```

Why?

Python executes `finally` before returning.

Flow:

```
return 10 requested

↓

execute finally

↓

return 10
```

---

# Important Trap

What happens here?

```python
id="finally_override"
def test():

    try:
        return 10

    finally:
        return 20


print(test())
```

Output:

```
20
```

Why?

Because `finally` return overrides the original return.

Avoid this pattern.

---

# Exception in `finally`

Example:

```python
id="finally_exception"
try:

    print("try")

finally:

    print("finally")

    raise Exception("Error")
```

Output:

```
try
finally
Exception
```

The exception from `finally` replaces previous flow.

---

# Difference Between `except` and `finally`

| Feature                     | except         | finally          |
| --------------------------- | -------------- | ---------------- |
| Purpose                     | Handle errors  | Cleanup          |
| Runs when exception occurs  | Yes            | Yes              |
| Runs when no exception      | No             | Yes              |
| Can access exception object | Yes            | No               |
| Usually contains            | Recovery logic | Resource release |

---

# Real Production Example: Database Connection

Bad:

```python
id="db_bad"
connection = db.connect()

result = query(connection)

connection.close()
```

If query fails:

```
connection.close()
```

never runs.

---

Better:

```python
id="db_good"
connection = None

try:

    connection = db.connect()

    result = query(connection)


except Exception as e:

    logger.error(e)


finally:

    if connection:
        connection.close()
```

---

# AI System Example

Suppose you call an LLM API:

```python
id="llm_example"
client = create_client()

try:

    response = client.generate(prompt)

except TimeoutError:

    retry_request()


finally:

    client.close()
```

Flow:

```
Request
   |
try
   |
LLM call
   |
----------------
|              |
Success      Failure
|              |
return       retry
 \            /
  \          /
    finally
       |
 release resources
```

---

# `finally` vs Context Manager (`with`)

Senior interview follow-up:

> If `finally` exists, why do we need `with`?

Because context managers make cleanup automatic.

Example:

Without:

```python
id="manual"
file=open("data.txt")

try:
    data=file.read()

finally:
    file.close()
```

With:

```python
id="with"
with open("data.txt") as file:

    data=file.read()
```

Python internally does:

```python
id="context_internal"
__enter__()

try:
    code

finally:
    __exit__()
```

---

# Interview Questions

## Q1. Does finally always execute?

Almost always.

It executes:

✅ No exception
✅ Exception handled
✅ Exception not handled

Exceptions:

* Process killed (`kill -9`)
* Python interpreter crash
* Power failure

---

## Q2. Can we have try without except?

Yes.

```python
id="try_finally"
try:
    do_work()

finally:
    cleanup()
```

Useful for cleanup only.

---

## Q3. Can we have except without finally?

Yes.

```python
id="only_except"
try:
    process()

except Exception:
    recover()
```

---

## Q4. When should you use finally?

Strong answer:

> Use `finally` for operations that must happen regardless of success or failure, such as releasing resources, closing connections, unlocking mutexes, and cleaning temporary state.

---

# Senior Python Interview Summary

```
try
 |
 |-- success --------|
 |                  |
 |-- exception --> except
                    |
                    |
                 finally
                    |
              cleanup
```

A senior-level answer:

> `except` is used for error recovery and handling exceptions, while `finally` is used for guaranteed cleanup operations. In production systems, `except` handles failures such as API errors, database exceptions, or model inference failures, while `finally` ensures resources like connections, files, locks, and sessions are released correctly.
