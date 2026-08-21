This is a very common Senior FastAPI interview question because it tests whether you know production-grade API architecture.

A good interview answer should cover:

1. What is Global Exception Handling?

2. Why do we need it?

3. FastAPI exception architecture.

4. Custom exceptions.

5. Production-level implementation.

6. Middleware (Logging, Authentication, Timing, Request ID, CORS).

7. Real-world project structure.

8. Complete production code.

9. Interview follow-up questions.

### 1. What is Global Exception Handling?

Global exception handling means handling all application errors from one central place, instead of writing `try...except` inside every API.

### Without Global Exception Handling

Python

Run

@router.get("/users/{id}")

async def get_user(id: int):

try:

user = await service.get_user(id)

if not user:

raise Exception("User not found")

return user

except Exception as e:

return {"error": str(e)}

Problems:

* Duplicate code.

* Different APIs return different error formats.

* Difficult to maintain.

* Bad for production.

### With Global Exception Handling

![](blob\:https://chatgpt.com/739b8d55-e31e-4c39-94fd-6984da4773d7)

Every error is handled from one place.

### 2. Why is it Important?

Imagine Amazon.

Millions of users call APIs.

Different APIs may fail because of:

* Database failure

* Authentication failure

* Validation error

* Payment failure

* Internal server error

You must return a consistent JSON response.

Example:

JSON

{

"success": false,

"error": {

"code": "USER_NOT_FOUND",

"message": "User does not exist"

}

}

Every API should return this format.

### 3. Production Project Structure

![Pythonのプロジェクトのディレクトリ構成のスタンダードが知りたい - ts0818のブログ](https://images.openai.com/static-rsc-4/62ye5iS7gUfeJUpN_B0psn_PTGUXgLtESgqr5zX61FEbCf5KA4NYFRhHP7dxd_MFNwRAyTD1TWL3EaYP7qkVNHu_A7ppGn0Oma4OhCruD3vWUeJ7_0XOiM-ASl3qbZ3ZzXajO5ZiZY9MGqmM_dQZnf668hvO6uyloDlqs0Hdpjs?purpose=inline)

![Deploy Single API from multiple APIs as a microservice](https://images.openai.com/static-rsc-4/6oyNrKLPd59Jjg901EFs5g0hGIJgqhFwJD0KjrMHQgbBCJ9H8qd3Tah3mMm35dWTP8lOxJ8FRWjtUmA3Zx1vHBmsbtedrdWDMe-SDgzafuKs9XeQ5-peJVadKHjjmWPhvumyomlpvdZhj1gLHOqL85CF3981FfuN4-OaQGiABRk?purpose=inline)

![Vlad Podgurschi
|
Web CNLearn - FastAPI 1 - Creating a project structure](https://images.openai.com/static-rsc-4/hLc5F7OT52x3gNkwjItoO43zBmKGErgKVzeyB4T6K1ywgK1rB-TURlx_6zYi05vp7bAMAgBW9W5Ieb8TXEyJzF_uInfeLZlzUgnJHnj5pNchHP0Ko4yrvjZNCUvHg4nnz8QINyDkNkwDezjFzq434pMxQkDBIqUVZy5cqDFdOEs?purpose=inline)

This is how senior AI engineers organize FastAPI projects.

### 4. FastAPI Exception Flow

![FastAPI Logging & Versioned API Best Practices for Python Developers](https://images.openai.com/static-rsc-4/FU--VxL0UsSQz2TJ6cdg4gRvyq9kLbHAb1Js8u6O1iTJNAR3ybuRZSSc_2zUvoqkuB3x2PgQ-nRnAnoCY50HuoETgULCg0yoY24u0q1qGwO65xpO5Vx2-orqS7nWSQoO7KYCs1otBn9KR5I6EMWWVwyyI9C8ZaPkeu1cvqDBims?purpose=inline)

![Building a Production-Ready Task Management API with FastAPI: Complete Architecture Guide - DEV Community](https://images.openai.com/static-rsc-4/w9cjK7ZTP_JMAerTlNqQHOmCcSpG4boauIRZYWvD0o1zb9_ni1qL33OA3pszYN3DcEo8U7WzwfMWF3oafx2M_zu-GGVmeTH8Pq6ik87ptqEGC1DtdlM6POGNuBN-i20OEetSf8ygL4qGQKSMokdwYX16oxdWq9OoCP9SLUzoIsU?purpose=inline)

![Building an MLOps Pipeline with MLflow & Scikit-Learn | by Faizulkhan | Medium](https://images.openai.com/static-rsc-4/KrvTc5FaGa6BOZD4P577ITkfYi7a2T8txh7olLg2uPUdBLJzfV5b87v76yf5yCVGrdt6D0szV8AZ_cxvYDSCAiXggr8wEGa2joKeiIJdNzKdULHP4y0JCBtuNj7RRZlCdj68pRS2ZM-r3gCrbPzt9o4CYrl4mRJctU5FNw_g1_w?purpose=inline)

Notice...

Exception can happen anywhere.

It automatically bubbles up.

### 5. Built-in FastAPI Exceptions

Example:

Python

Run

from fastapi import HTTPException

raise HTTPException(

status_code=404,

detail="User not found"

)

Response:

JSON

{

"detail":"User not found"

}

But this is not enough in enterprise projects.

### 6. Create Custom Exceptions

Instead of using `HTTPException` everywhere...

Create your own.

### custom_exceptions.py

Python

Run

class UserNotFoundException(Exception):

def __init__(self, user_id: int):

self.user_id = user_id

Another example...

Python

Run

class InvalidTokenException(Exception):

pass

Another...

Python

Run

class PaymentFailedException(Exception):

pass

Now your business logic becomes clean.

### 7. Service Layer

Instead of returning HTTP responses...

Raise business exceptions.

Python

Run

class UserService:

async def get_user(self, user_id: int):

user = await repository.get(user_id)

if not user:

raise UserNotFoundException(user_id)

return user

Much cleaner.

### 8. Global Exception Handler

Now register handlers.

Python

Run

from fastapi import Request

from fastapi.responses import JSONResponse

@app.exception_handler(UserNotFoundException)

async def user_not_found(

request: Request,

exc: UserNotFoundException

):

return JSONResponse(

status_code=404,

content={

"success": False,

"error":{

"code":"USER_NOT_FOUND",

"message":f"User {exc.user_id} not found"

}

}

)

Response:

JSON

{

"success":false,

"error":{

"code":"USER_NOT_FOUND",

"message":"User 15 not found"

}

}

### 9. Handle Validation Errors

Suppose user sends:

JSON

{

"email":"wrong-email"

}

Pydantic automatically throws validation error.

Default response:

JSON

{

"detail":[...]

}

Enterprise companies usually customize it.

Example:

Python

Run

from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)

async def validation_handler(

request,

exc

):

return JSONResponse(

status_code=422,

content={

"success":False,

"message":"Validation Failed",

"errors":exc.errors()

}

)

Example response:

![How to validate an API request in laravel | Medium](https://images.openai.com/static-rsc-4/04ploxXtiYxXasbrZ0OeHrdMyewtW8zR-uLhAkkY-TxLN38b0_Iu6aisYUrK1xmDQQeSZzrXRinjDINzVERtrp9x-b0dhtR8j7YstEM7-Ff3LWFUt19GmDCxJtiWCYMx0YX6gfbh5Tj87TOaZnwwg0deIatMcUJAZXFXuwktDxk?purpose=inline)

![JOI - API schema validation](https://images.openai.com/static-rsc-4/6fEL7u9u0J1BduE6WdMCuqrCGlwe4hhTeqlkAX3NsUuQp_7MLbB-dO8XRe8YOeTnDuXO5q8I4VslDfDIQkPdE4u4VYVYZwxWkzbAEEd9EOrrsimlC3tG3d-zoEYyucvr28yyjGJN0NkpmY0_bfi6UQPDmNJGFypKGgjdq8YjtZY?purpose=inline)

![Enhancing Your .NET Core Application’s Data Validation with FluentValidation | by Umair Rasheed | Medium](https://images.openai.com/static-rsc-4/hXIyGqX4-TM3PK3gte0L3Cd_ES3fh81QkRKNCwFlvxtAEoJ8nFM-LqmMABvSJg0FCUuO3qN6I3fHvOVYQmpdPH88Ti3C2u5IcZO6q-bui3vPFQcbt9zp5UTZNLHC6vnU-5MZunxfrqChB49f35rS3mWMWIp6M50hNtSwmMVzIHs?purpose=inline)

### 10. Handle Database Errors

Suppose PostgreSQL is down.

![The Database Connection Pool That Brought Down Black Friday | by Devrim Ozcay | Javarevisited | Medium](https://images.openai.com/static-rsc-4/W3rKbAsVsJ7GfFtbBcpl_B9KjLshVi3MDdElrx79M-yw0sUJYFKFHqPaHSWqKtlj-lT4Ci0g9I1mniIdACQRKeDdvu0Ttdzs5lxyUoePpYfd6Ys4XdDAFHZxw4K6Lwo5KYPPoMKO6eC0pJUPPTs5aRz85A_20mMeBV5G2yMtRqU?purpose=inline)

![Error 500 Coolify DB](https://images.openai.com/static-rsc-4/G44zNOIyfTZGm96lFsOzHA60ghOr1wRkMflmwAtimB4OLCrmynepa935GQyZuQJ0dkjJQtkAG4dwBv0zs-Tb_PLXGtYpjd4_ZlZ3LkaNbYzr3KVomZE3ae08wYQMgyUtZVcgHsbR8Sg_SjVxurrAxdlMLC16kHPhlCrCyotnpuM?purpose=inline)

![What Happens If Your Production Database Crashes? - DEV Community](https://images.openai.com/static-rsc-4/1z5RIIrU56EJB9asFkSMUwlH6cRftbhoG6py5_Tbo2e4ljEThob7pc1TJBb0vmnYPKA_h2CXyfYhSxmpvrQoPq771Cjk300Vm6FCOWw5vSga7VQ-POOGDeeZJlpwF_325C0cn15vB41xmXQ92hSDRPo6JjPaDLTiIastQEz0kyU?purpose=inline)

Example:

Python

Run

from sqlalchemy.exc import SQLAlchemyError

@app.exception_handler(SQLAlchemyError)

async def database_exception(

request,

exc

):

return JSONResponse(

status_code=500,

content={

"success":False,

"message":"Database Error"

}

)

Never expose SQL query to users.

Wrong ❌

JSON

{

"error":"SELECT * FROM users...."

}

Correct ✅

JSON

{

"message":"Database temporarily unavailable"

}

### 11. Catch Unexpected Errors

Always have one final handler.

Python

Run

@app.exception_handler(Exception)

async def global_exception(

request,

exc

):

return JSONResponse(

status_code=500,

content={

"success":False,

"message":"Internal Server Error"

}

)

This catches everything.

### 12. Production Error Response

Instead of this...

![reactjs - How to send json data from the backend (express.js) to the frontend (react) after res.status(404)? - Stack Overflow](https://images.openai.com/static-rsc-4/zQghw6NkcGGQFTHvlWHM0iQ4nGEDbIJC1gdv8pSn1i8ya6Cjcqo77Y0sXxa3uclQPCj9UVQdE-8j7FEJ1yiPeC64D47lAlJ5VR8Yd76pdaMgw7AgTj2HzByEEDkfQ1gkvcQgoS9KyBuTVcOBGSSByvbAivdDFT6TC5kgn7NrADk?purpose=inline)

![Unificar las respuestas de las API](https://images.openai.com/static-rsc-4/FlpCfmO447TT2xY3ML8nwI-ManUAxNHLpFGlGvkvfFuPoZW6KOH4U7NEU0Idm2_Wm-Hsnh62W-qEFjgFrsCnoXao67-cPC8_qpaPwYDflQQYGJOWcjIdDy071hxpHov3EYDdAHpPtHYUxmbY3ntnN4E1lShkvC-z6YA_OC0FQDE?purpose=inline)

![Exception Handling and Validation in Spring boot - DEV Community](https://images.openai.com/static-rsc-4/7OQuqI2zYaDuO_RC4KQ_JJw_O7Ww6vuZH_2Mb2KFZ3MO5hVnX0nT9qc4_pW_oYEMweKVcXr7VekMm8Xh6P3Rurh7rbRxQBiCY8LXvBiVRXW9pnXeHHAHoRgX0KRYaNvXTFVrbhlgM6_RoaensP2LZ36D5ACWzECFJ0ABsqKBWaY?purpose=inline)

Use this...

![Unificar las respuestas de las API](https://images.openai.com/static-rsc-4/FlpCfmO447TT2xY3ML8nwI-ManUAxNHLpFGlGvkvfFuPoZW6KOH4U7NEU0Idm2_Wm-Hsnh62W-qEFjgFrsCnoXao67-cPC8_qpaPwYDflQQYGJOWcjIdDy071hxpHov3EYDdAHpPtHYUxmbY3ntnN4E1lShkvC-z6YA_OC0FQDE?purpose=inline)

![Proper Java Exception Handling](https://images.openai.com/static-rsc-4/ir6IaYSmkttzYO3DKdgfFKUNkI6TU29pxBcLqb40fyMWH1CdZ18Sg_2maJa-SIeww3unkcD-BuU-aSHqQxSObnFfU1Lis2qvRMjEBsAeAOexxunLOIygd86l7E3omlxRaa9yfE_QLZeZAN8LbyXxmwr34X3F8rpKYZGnZkr9bNU?purpose=inline)

Much better.

### 13. Logging Exceptions

Always log.

Python

Run

import logging

logger = logging.getLogger(__name__)

Inside handler...

Python

Run

logger.exception(exc)

Log example:

### 14. What is Middleware?

Middleware runs before and after every request.

Think of it like a security gate.

![Middleware - FastAPI Beyond CRUD](https://images.openai.com/static-rsc-4/Nqh__dGjPyCIeI06WS8MHQrD6fnyfSwlggnN_e01AfwhGtMlH99U9-RHtbo050jlslgihiCUfppQmLKFG6Ye0X9lxstR2h8YKQM9VEUmguqxC2pmISWlheBamxDVtB0Hl5ZvH4zqlt3Gbj3R1PyBscKzji5xA8oJohqR0exiuVE?purpose=inline)

![FastAPI Logging & Versioned API Best Practices for Python Developers](https://images.openai.com/static-rsc-4/YxE8Z_MPS_T6IcmgoHFvJdRBAPZxNPqAtGrEHutdTPnSWb8bblrnVb2Z2ViWvNdnnEi0BZfgM7Gb9HlQ48Ff0XfQF7xrIoQW-mRq3AWCtU5hZ7TArvr2vsL13MZ-0-GFfem-lpxhcnK33jNg53-i5xjZiVnzxHCLTdWuLlR47R8?purpose=inline)

![FastAPI Logging & Versioned API Best Practices for Python Developers](https://images.openai.com/static-rsc-4/FU--VxL0UsSQz2TJ6cdg4gRvyq9kLbHAb1Js8u6O1iTJNAR3ybuRZSSc_2zUvoqkuB3x2PgQ-nRnAnoCY50HuoETgULCg0yoY24u0q1qGwO65xpO5Vx2-orqS7nWSQoO7KYCs1otBn9KR5I6EMWWVwyyI9C8ZaPkeu1cvqDBims?purpose=inline)

4

### 15. Middleware Execution Flow

![FastApi background tasks — but better. | by Snir Orlanczyk | Medium](https://images.openai.com/static-rsc-4/-4UaH8wf5q4aN02ikVsJybEPukrubJtIdWVsw6OC11qP95fRdLhFwSMwx-nC6GUXRKdG-3eqx0jB1vHgoxqblbkGaKb0WTz6vWbRSyNEmtEO15E2bDAuA0lZuMmyoFX7KnbaJXADA49_tOWq0C8mkilfugzg24FCxuyCOZrtukM?purpose=inline)

![Python FastAPI middleware to modify request and response - DEV Community](https://images.openai.com/static-rsc-4/VEOh1l_-TeXd3XO3YzdNr7iQOdiG9V6ITbYwZilf-QN8_m3jYpOGmwhGStiRpMatlpFXgmK_oSjl17Z-XwcKojilUp_dtRqowb5shftg0zwZDB_2viGBnOfTZlaR1fpj12v7LZuzi2pq7sPTv63xTkfwh4DAK5SrstOsUaxeZSA?purpose=inline)

![FastAPI Logging & Versioned API Best Practices for Python Developers](https://images.openai.com/static-rsc-4/YxE8Z_MPS_T6IcmgoHFvJdRBAPZxNPqAtGrEHutdTPnSWb8bblrnVb2Z2ViWvNdnnEi0BZfgM7Gb9HlQ48Ff0XfQF7xrIoQW-mRq3AWCtU5hZ7TArvr2vsL13MZ-0-GFfem-lpxhcnK33jNg53-i5xjZiVnzxHCLTdWuLlR47R8?purpose=inline)

4

### 16. Logging Middleware

Production example:

Python

Run

import time

@app.middleware("http")

async def logging_middleware(

request,

call_next

):

start = time.time()

response = await call_next(request)

duration = time.time() - start

print(duration)

return response

### Flow

![FastAPI Logging & Versioned API Best Practices for Python Developers](https://images.openai.com/static-rsc-4/YxE8Z_MPS_T6IcmgoHFvJdRBAPZxNPqAtGrEHutdTPnSWb8bblrnVb2Z2ViWvNdnnEi0BZfgM7Gb9HlQ48Ff0XfQF7xrIoQW-mRq3AWCtU5hZ7TArvr2vsL13MZ-0-GFfem-lpxhcnK33jNg53-i5xjZiVnzxHCLTdWuLlR47R8?purpose=inline)

![Introduction To Backend Logging](https://images.openai.com/static-rsc-4/w7weUj0v2I1nUjwjckKZRR0KXR7gJlsked2ayOFkryAYgPN9RfX1NyHu0dpowjGyAXUfV37tkKO_ycGShvAhm4LzZvo0QK3tfVyUWKGaSmEDEN4CHK4cu2MHycNEocM7_iEmQeueJg89Ih9XwKbvqKfFk-hwONupAcLKoqBKQAs?purpose=inline)

![Creating a Lightweight Logging Layer That Doesn’t Suck | by Tanveesh Singh | Medium](https://images.openai.com/static-rsc-4/P6kAyEO5ONyTUATJVufQ9n41ZaQD7IIhFs4lBJiRueyOtftYuApauhIm-mxM3NZgnkjH83j7-X0bjIG75B7_3AIku159GwvMei_ezoEk3SAFg_phlGzdaEzJJW-yza2gc_cfXB6YiDX5lLf14P46xFQXPuRWrAYNl6YDyknTMfY?purpose=inline)

4

### 17. Production Logging Middleware

Python

Run

import uuid

import time

import logging

logger = logging.getLogger(__name__)

@app.middleware("http")

async def logger_middleware(

request,

call_next

):

request_id = str(uuid.uuid4())

start = time.time()

response = await call_next(request)

duration = time.time() - start

logger.info(

f"{request.method}"

f"{request.url.path}"

f"{response.status_code}"

Response header:

![腾讯云可观测平台 Aegis SDK 支持获取请求头和返回头\_腾](https://images.openai.com/static-rsc-4/wQLjyQ0GsGRZSJYCtw1q-5oc8C_fneQcWRnA7aTfWJ8wFRPAB_y776fEAVHMbpD-SKSZOyoiOUptfHr1wNE9ITx98bX3OfeSUYl7dyaUda-nPOl4EODi_10iRXRemzB9GIUABDj51oBAOgYAh_LXBt_dGS-E2BQki2QA0-PEOgg?purpose=inline)

![Getting help • Debug with Chrome DevTools • Palantir](https://images.openai.com/static-rsc-4/xCmbTN9XAuBXqM0VDSFp7zl828wLPjoxWlY691sUCrxVtfcgaCZzjiuNQeASpbAiff0BLjO4vKptBK2syxdMrXwZ3Y9Kq6Q474crfHxpmi_T7WBLblYxgLqyRR_1WcQ_IJvybF3gL9NJPJGV0slJJ4P5FQ--3ovAmmSb5Wjy23I?purpose=inline)

### 18. Request ID Middleware

Why?

Suppose 10,000 requests arrive.

How do you identify one request?

Use Request ID.

![Logging, Monitoring & Observability — The Missing Pillars of Scalable System Design | by Gyanaa Vaibhav | Medium](https://images.openai.com/static-rsc-4/u_2zDZ9KMitFxW5vrYIAoIPe9QwOp2I0AoBZrEzB44w3egmydGOy3GIPzdCl8jQljEQ3M6Rj_xacm9_h8DcKIg8eOgNJh_K0Pf0VpH_P007ZcXXe7PQe_K3pZR4jZBAHIe8oIlZDzUmeMCiSUf5-rcBRasHms9vajodD6BC8CkI?purpose=inline)

![Empowering Developers to Achieve Microservices Observability on Kubernetes with Tracestore, OPA, Flagger & Custom Metrics - DEV Community](https://images.openai.com/static-rsc-4/b7ygWcMqLTvnqhvKMvrvTcqCGdYoe0h7I5DHdC7__tTttPSbPMGLSenJV81iWbYROBzYT6yo3sUx6vlTnawiRUf23frAIoAseR0N7PVpNxFGFn1K3-vyi2rhwnoZVrkYalcobsaMLEggP6wvfo73lpIfhuJnA9W8zHLQ9-Ww_lk?purpose=inline)

![What is Distributed Tracing? Full guide](https://images.openai.com/static-rsc-4/529RUaF_qMOmePBdyNKGvZxBo0DfzxunNgEFUDi_Ipi5o6DoAyuggbuHf8lWMyD_Vpu3Ot6cu0_7IrzCUGlcWfWcoEzV3oNiAv0yRM1W9PdHkc5ExIhck2GCR0_vWRyeCBzxw2nQUdsqSg2yyUG5fESdNRFTKf15kHKlDTbwSe4?purpose=inline)

5

### 19. Authentication Middleware

Instead of checking JWT inside every API...

Use middleware.

Python

Run

@app.middleware("http")

async def auth(

request,

call_next

):

token = request.headers.get("Authorization")

if not token:

return JSONResponse(

status_code=401,

content={

"message":"Unauthorized"

}

)

return await call_next(request)

### 20. CORS Middleware

Very common interview question.

![Testing FastAPI CORS settings • vim, git, aws and other three-letter words](https://images.openai.com/static-rsc-4/AXhl4DFdqxCtOJ-YzzdOn2TUANH7dyXI57bVmq7x-V5e5ABL-PAU-_0aKEe64oz3oL9dZOSfeDEIudRvFUjvsuliBBZB1eQtwedFFZka_F9mNlus8nZVu3oZoy6HQHAIQdNVZhpKz_xQ06MUq1IbKcCnKelPZaJD-bwYqLSCl0c?purpose=inline)

![Deep dive in CORS: History, how it works, and best practices · Ilija Eftimov 👨‍🚀](https://images.openai.com/static-rsc-4/jRL4SqS48JUzHMM4dIK2Yfp2NmN4DjbnKor_pMdrB9ZYuZwMhmQdmYHo7B2pHwlhKl_FJIUqxcIWn8zi-GpfCRMr_9PRs1bQRpk0pmZXvIZRC45FyIw1XGGs-8f3iEW7BjfqvPdKbgC3ERHCTD4AH-sW9IXuMy3ibnTnHlWPKSk?purpose=inline)

![CORS预检请求详谈 - wonyun - 博客园](https://images.openai.com/static-rsc-4/5Xc9-DxzJApdKgpOob1rLlUdJCYpqWwJvdHyGZAPgZ4uUVu_RRe85RH-a33HSreskLVX6fwxwAVSgIfrBFgQ7kBu41DM-3PcRRQQOInMeL_jmT0XsGUWftjX02-ZsXFD_oPPt7D45tOcypNyNYw1GwZbVqmNzHPgJPmBvKmTQvY?purpose=inline)

Implementation:

Python

Run

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(

CORSMiddleware,

allow_origins=["*"],

allow_methods=["*"],

allow_headers=["*"]

)

### 21. Timing Middleware

Measure API latency.

![What’s the difference between API Latency and API Response Time? - DEV Community](https://images.openai.com/static-rsc-4/Gr7XAo-8fb5uFTIg_zrq274_10ct5XLYN4H8nfVU44cI7Aaa3emFHKPFLOZn34mA7Kr8ErMXXRGO9UHHKsuomh6nDLYdHV9lIR8NZRsn8xB-E5IIJDp4Y53swHeWYwNjQ-ie8ZhUGNQNVNA2UTLbdhUsKkZZLVO-nZjBHsQ13F4?purpose=inline)

![Free Atlas instance performance vs the paid instance - MongoDB Atlas - MongoDB Community Hub](https://images.openai.com/static-rsc-4/hRPjGzx0InP__sJ2LOs9SdlomIrsGag3jGtikvSziB-QEt8ACXU6KCgsOsqAF9FPaWDg4DrsY1LWT8ODw6DBASeCB93YToJM62grFwvP7SFsXeU7JHSw-jSpZTX8ySYnneyGqZ4D-33v1u0dGAV8tU3mmFq3LWpno-iiEXNaVDw?purpose=inline)

![Designing High-Throughput APIs for Peak Retail Traffic: Lessons from Black Friday at Scale | by Renjith Raj | Apr, 2026 | Medium](https://images.openai.com/static-rsc-4/2mZC91gBTA_d0IuYwQVVzf3ROe9KuzPo4MN2DWsou6pIT88VuCl1VTro0-UT1O0LMin3p5Frd4W7kLGy2xsCCiVhDB77Sx-jRrs3k53Q4naapw_qZo7EGM9HK2KecVTTbeknwVlKsAeRCze4c6cQJxcSdFsFyzije-htZPOrSuU?purpose=inline)

Code:

Python

Run

import time

@app.middleware("http")

async def timer(

request,

call_next

):

start = time.time()

response = await call_next(request)

response.headers["X-Time"] = str(time.time()-start)

return response

### 22. Rate Limiting Middleware

Example:

![What is API Rate Limiting? Prevent Abuse & Overload](https://images.openai.com/static-rsc-4/PETRNvJQlneCHH5Y9AzHz_4ujSuYPMnKfBiksaB4rIbp7cmg4ngPF4VgFDs4vFMaBmxxZWfveoA0-BLudz0bXVIdraUxXrHDYqEQqR1AmxSIQEIi-ai6bxILV2JqdWXsg9jKOKIL8y0g3JC4kqs2aQySHPc5tt4xX5tgJs8b6Xs?purpose=inline)

![Rate Limiter System Design (Token Bucket Explained) · InterviewCrafted](https://images.openai.com/static-rsc-4/WsaVNC4eYePjWHS4RDKpoEzHEK3OZV6ABbOCDE_Rybcrm_HyUg28fSMX8aYuaLTZNOnJT96j8U1dgJyBgKzr1-aSf-C6vkgqa7ll41ZRZmBcjrY51JH2y6SgJo9Z--JUPz0-ul0-ifm2iv1WAjFpx-2JzSGjRqrYq6XJd-k6xFE?purpose=inline)

![Rate Limiting | Course](https://images.openai.com/static-rsc-4/xoOya-2bwzmrctMxTKNhB526ieWI4cwGaUYiwfY7CEQtRMd11SUjBnRb9u3JJZHFdy-5BDBM-nVlBrUgTBo5mAo5emwrtENhzt6jw1N55AkC-2Yg7WxVFbkftgOxxreuAgfO9I8HDk6AutOmm3KZOXk-pc9VcMafJn-k_qUFrkE?purpose=inline)

4

Usually implemented using Redis.

### 23. AI Project Middleware

In LLM applications...

![8 Middleware Layers Between Your Agent and Production | by kumaran srinivasan | Feb, 2026 | Medium](https://images.openai.com/static-rsc-4/Uf4IUKJ_YzU-FrX3aw2f_VfFDD9SgtG0AIr156aAlqk1rmki1Vm_5MGLdb5jIkKae4z42V4AYMNjdtGKtQWqn8wnXC-IljUzhCi2xxge3KfgNbEZSS2CKoEkqcTjvuyhKO01dF7ebQ4SpO4rQ730QA6SEoEz1PepBIBJQJ1LdjY?purpose=inline)

![Gateways, Guardrails, and GenAI Models | by Manish Dewan | Expedia Group Technology | Medium](https://images.openai.com/static-rsc-4/Qjzv7Af4MCNjUaEuhJyA317bpaE53kAPz6X_DKW0ejA4LbQojP8UVFVGwpsKmUFkHVMyM4FSeUUBUImsLfhi1pRTLpg2L-bvJLpckgqX1Wre1JGjKU4bki48aB8gWbQbKAYqPqDOH3f5TnjYZJEmYwMYZzvvDGuZtYOmA_CQ2jw?purpose=inline)

![Do You Actually Need an AI Gateway? (And When a Simple LLM Wrapper Isn't Enough) - DEV Community](https://images.openai.com/static-rsc-4/AIdSt4xPvhraF7wanqY8GAAKtEDgOP1FGNtO_ewooODUk6r9spYuPPDc82kuEqQRz2h2Lyh-gbHIJb_QkgdeoqqG6k7ePlDk53kNjEMlJtX2Q-iZTF3QhmDbesb9BnJlG_5KfNMGApozcDKHVwI6xoeGM8avBl6mId9KeYrgQwE?purpose=inline)

5

### 24. Middleware vs Dependency Injection

![Rate Limiting with FastAPI: An In-Depth Guide | by Dhruv Patel | Medium](https://images.openai.com/static-rsc-4/VYetIIMlZtDP4tX1H3QxEFxh3xdYo2TY51DJg4FE7kzedrGEPxGnyC3GLJ3Fwh_JbPtr23om9BzvZEb7229AUZzgXFnfYZrU7ZsUiZRdGie-RWLEgBjsZHbWI-9UIHhSQgKTAagsEIS_D0GqmeVULlyPqxWqzOvdLyIt6borfYM?purpose=inline)

![History of FastAPI – From Type Hints to Modern Python APIs | Thirdy Gayares](https://images.openai.com/static-rsc-4/TN3CKLbi2yNVmOmIYB_HMk93TAOOUYrfTiZXzKN84oiaj7BJ7SF40bsmhdK0b3eibEiX-1ASr2gpzNgSehjBjCp2lnMJsruzOXGKuo6JqR-mBcQ9s8qe-PlkfS6JOi-JxBSPEqiU8ytTVv__Yonf7qoWbvyfXmF2XWwZ-gT-xrs?purpose=inline)

![FastApi background tasks — but better. | by Snir Orlanczyk | Medium](https://images.openai.com/static-rsc-4/-4UaH8wf5q4aN02ikVsJybEPukrubJtIdWVsw6OC11qP95fRdLhFwSMwx-nC6GUXRKdG-3eqx0jB1vHgoxqblbkGaKb0WTz6vWbRSyNEmtEO15E2bDAuA0lZuMmyoFX7KnbaJXADA49_tOWq0C8mkilfugzg24FCxuyCOZrtukM?purpose=inline)

|
Middleware

|

Depends()

|
| --- | --- |
|

Runs for every request

|

Runs only for selected APIs

|
|

Logging

|

Database session

|
|

Request ID

|

Authentication

|
|

Timing

|

Current User

|
|

CORS

|

Repository Injection

|

### 25. Real World Middleware Stack

![Inside Open WebUI: How Browser Workers Bring Python, Plots, and Speech to Chat | by alex buzunov | Medium](https://images.openai.com/static-rsc-4/Gf9yzvF3qaKOsqi8vAdo3k3_Qpzn7O9_xt3jsdmYTW3eovgsVVo9pyyFByI10ulT-bt9JKjnDhsxh7ejeVBvFg8KmKY5qQ_nXZiNbjCptZG8YbCl5EMJsjBwIE2VN5mYkgiqPYKvBg_Cd6jGVZlGEGLVefdGNqefQGtItk8CArA?purpose=inline)

![5 FastAPI Middleware Designs That Add Value, Not Delay | by Velorum | Medium](https://images.openai.com/static-rsc-4/TT_nlVTX_8LI_GUk8ck2DH6CfknKHitDxZg9Iq2QMMA0jqDJqR0HdMMUhUstX8mkrtkQIcpINOo-QkKNw8uqRWk-9AM3gcIJQ4QLAKnW7IV9K46KyND6Uwdl198cDFTGZRGiIYrizOB18KC-42NA40R5AGXUFY14Wrl1rigAncA?purpose=inline)

![How to Secure Your API Against Unauthorized Requests - DEV Community](https://images.openai.com/static-rsc-4/TPiAFG9u2N2O3S2f2IIpAA6peeBnQLBYjRzIT6J71e3e5lkSVp5AaO58VApQFtjuqIEy6VeF1u5n2_JE6KFmP08aCKjDJCsZWM207KmmRROYy_FriCpwMGYO__Rdg2vVoIvRAr1cJ8EtHFkcP-MwcGzenx_3JZasQauT6fNdKhQ?purpose=inline)

5

### 26. Production FastAPI Application

Python

Run

app = FastAPI()

app.add_middleware(CORSMiddleware)

app.add_middleware(GZipMiddleware)

app.middleware("http")(logging_middleware)

app.middleware("http")(request_id_middleware)

app.middleware("http")(timing_middleware)

app.include_router(user_router)

### 27. Complete Enterprise Flow

![Building a Production-Ready Task Management API with FastAPI: Complete Architecture Guide - DEV Community](https://images.openai.com/static-rsc-4/w9cjK7ZTP_JMAerTlNqQHOmCcSpG4boauIRZYWvD0o1zb9_ni1qL33OA3pszYN3DcEo8U7WzwfMWF3oafx2M_zu-GGVmeTH8Pq6ik87ptqEGC1DtdlM6POGNuBN-i20OEetSf8ygL4qGQKSMokdwYX16oxdWq9OoCP9SLUzoIsU?purpose=inline)

![FastAPI Logging & Versioned API Best Practices for Python Developers](https://images.openai.com/static-rsc-4/FU--VxL0UsSQz2TJ6cdg4gRvyq9kLbHAb1Js8u6O1iTJNAR3ybuRZSSc_2zUvoqkuB3x2PgQ-nRnAnoCY50HuoETgULCg0yoY24u0q1qGwO65xpO5Vx2-orqS7nWSQoO7KYCs1otBn9KR5I6EMWWVwyyI9C8ZaPkeu1cvqDBims?purpose=inline)

![Portfolio Project Blog post. Project name: Social media API backend | by Cappy Alemayehu | Medium](https://images.openai.com/static-rsc-4/XnAdYyGqw6LbGbX4wxLbrvp5JP4z5T23TDhTM--QQ74LG4fEhHTO9-GkS9QeUtwg4BUqGpezahoMCUrFzmdCe35xUcKnEuJQFQp0LPa182dnb2QuSGRCOdsY04QdjumrHoGXBmfvLISB9xY2mOzGtZdyTKSedDARPXdQqzNMJaE?purpose=inline)

4

### 28. Real AI Project Example

Imagine a RAG application.

User uploads PDF.

![From Prototype to Production: Building a Reliable RAG API with FastAPI + ChromaDB - DEV Community](https://images.openai.com/static-rsc-4/MwBAva3GXk-Dk6zJFXTct7AzIx2nZrK73wyq9D59WZrHj5B_lqZJNApZim39NVfypMtVaIFYzLZ-prreDxAGJb7ly2gpF5ZhlYsON5GDIk8Te7fVGTFtQeFFXW2hg7D-ToTOeqjD7jd5mTC4b7kYq3SMTJ8_XmOgsv6Cqc_t87Q?purpose=inline)

![Designing a Local Retrieval-Augmented Generation (RAG) System with FastAPI, ChromaDB, and Ollama | by Shubhanjali Singh | Jan, 2026 | Medium](https://images.openai.com/static-rsc-4/y-HqiiJJ8xrG6FmlyBYOIVRPshraMsGjR8H3YAN5ASBmMSqfaBnNhzwpDN0mEKzXofPk6VNieOvEu5VZhoKcKW6SL__0aI4mYBYQxmkCkXW_fv1_9ZjBNhVksotTLRfa2aTp6S89k7NPlK6170wdkZ9i7f5ekk6TqQBrkWgLBkY?purpose=inline)

![Mohamed Aadhil Imam | AI Engineer](https://images.openai.com/static-rsc-4/pOuak-2E84wfLxclpj1KDKlgxIVH0inTV4gyuKJITOT-r2bvMvMppELk8o2kgRhCP3FXNyJ92vE4zO7BagbogCcTMU8GppLe5ZfeTZVXy1ij3a0c0xJrkS3INVRy4M7861tyGKwZ2LKyBfQdLTIsLOmAruHQjxgquKY2UltIPqE?purpose=inline)

5

Middleware does:

* Authentication

* Request ID

* Logging

* PII masking

* Rate limiting

Exception handler catches:

* PDF parsing error

* Embedding failure

* Vector DB failure

* LLM timeout

### 29. Production Error Handling (LLM Project)

Example response:

JSON

{

"success":false,

"request_id":"abc123",

"error":{

"code":"LLM_TIMEOUT",

"message":"LLM response timed out"

}

}

### 30. Senior Engineer Best Practices

![From Script 📜 to Service⚙️️: Building a Real Mini API with FastAPI | by Neural Curiosity | Mar, 2026 | Medium](https://images.openai.com/static-rsc-4/7WJk5vnWxEC5p11uKq5xzMeB2MTToal4D6LP-IoVs741w2nfmeHWn6-HUM8XqkI9ib2RHy9HyCsSgfJ20jU9D6pjRn8ns9ta6hKgYBMFSfe3L0la-oHUQzrGww_d1uTQqZycMh2j1oU8uwnPeLhm1sjoxiNBfehHFKkdWM4DXOU?purpose=inline)

![FastAPI Error Handling: Types, Methods, and Best Practices - Honeybadger Developer Blog](https://images.openai.com/static-rsc-4/dlJ5yLCZprpWY1tvZ2FFPj6mdH_kdRuWUnclXkzCir8t1EoaYbGrHEZtX0uVU7pVIEuoihATTURU9L0gcuP3D1EPi_8SMraptEF6kftyw_klmURghvCV3Yl8I9QgGyxiKC5RqxLVjggkwVmwIywdPcPU2VO6u5v442Ka66vmGHQ?purpose=inline)

![Understanding User Registration: Email Agent Series - Part 1 - DEV Community](https://images.openai.com/static-rsc-4/-lnSFJys8csNsINB6s1MM_qXR4UbfFsH330nLqxH1vViPnIt9mdnTE-aNd8w1C72-n8gnHay-Y_Q-n0J3lfxm_czPwmmqd3uKUwGG8VDjaBtV5dhEX_MhIrd1K8V3I6fVqI0a-c1xGpzXNFGaU0kI0R4RVh8RclDdaPWiaYGC2E?purpose=inline)

6

Always do these:

|
Best Practice

|

Why

|
| --- | --- |
|

Custom exceptions

|

Clean business logic

|
|

Global handlers

|

Centralized errors

|
|

Request ID

|

Debugging

|
|

Structured logging

|

Monitoring

|
|

Validation handler

|

Better client response

|
|

DB exception handler

|

Hide SQL errors

|
|

Catch-all exception

|

Prevent server crash

|
|

Middleware

|

Cross-cutting concerns

|

### 31. Interview Questions They Usually Ask Next

### Q1. Middleware vs Dependency?

> Middleware runs before/after every request. Dependencies run only for APIs that declare them using `Depends()`.

### Q2. Can middleware modify requests?

Yes.

Example:

Python

Run

request.state.user = current_user

### Q3. Can middleware modify responses?

Yes.

Example:

Python

Run

response.headers["X-Time"]="100ms"

### Q4. Multiple middleware execution order?

![Middleware • RestRserve](https://images.openai.com/static-rsc-4/cQah58feVVPBMfR43KH8_srXXGogUDmyIwrs8_085K5xVJTSwuuHs3zoP4i99BRveGGG1xxGpZhGbLDo5lGOH6YgmmTDdcJUqFtmDr1IRb1gpPuMAEPtp9HugmI8n4kjFRkHcq5YhZ6ZfFCgLUaWjpRLydnLwz8MPQbr8-r75sM?purpose=inline)

![浅析 Dapr 里的云计算设计模式 - 张善友 - 博客园](https://images.openai.com/static-rsc-4/vbRzDuu7uBGrVd77qpC5EIQ4iVeuxJz7u1mEBlB6S8FDZxPPaggVkfa-9JrnpkgiIC7sYNWdCuIXv-_dE9d0zPu3dLUCg3JUQ9ANeQCTUBaBoO8GekUJ1l8VVnQwhE6x6QpydMxryMz1vvRLTe_UyfZtoiNAnKCj4ozOFvLxjAc?purpose=inline)

![What is Middleware?. In the world of web development, the… | by Zahid Çakıcı | Code\&Beyond | Medium](https://images.openai.com/static-rsc-4/Sx9zqBdIzNPVKiS7L5le6VFjidtMNvEsy0IM5y7r_HjjUPatiL5w9BmUDgmZIvRstRGs7bkJ0RkPQ13-KhYuq9O4H-oOEkxLMwybZnSs-TG0IC6mBxH49QSx8SLbbfYqn9WdUFsSqGyozdB7gkC6UZTtY9taRHp9jUQ7hqehMbQ?purpose=inline)

Request goes top → bottom.

Response comes back bottom → top.

### Q5. Why use request.state?

To share data across the request lifecycle.

Example:

Python

Run

request.state.request_id

Accessible inside routers, services, and exception handlers.

### 32. Production-Level Enterprise Template (Recommended)

This is the structure used in large companies (EY, Microsoft, Amazon, Stripe, etc.):

![Building a Production-Ready Task Management API with FastAPI: Complete Architecture Guide - DEV Community](https://images.openai.com/static-rsc-4/w9cjK7ZTP_JMAerTlNqQHOmCcSpG4boauIRZYWvD0o1zb9_ni1qL33OA3pszYN3DcEo8U7WzwfMWF3oafx2M_zu-GGVmeTH8Pq6ik87ptqEGC1DtdlM6POGNuBN-i20OEetSf8ygL4qGQKSMokdwYX16oxdWq9OoCP9SLUzoIsU?purpose=inline)

![Pythonのプロジェクトのディレクトリ構成のスタンダードが知りたい - ts0818のブログ](https://images.openai.com/static-rsc-4/62ye5iS7gUfeJUpN_B0psn_PTGUXgLtESgqr5zX61FEbCf5KA4NYFRhHP7dxd_MFNwRAyTD1TWL3EaYP7qkVNHu_A7ppGn0Oma4OhCruD3vWUeJ7_0XOiM-ASl3qbZ3ZzXajO5ZiZY9MGqmM_dQZnf668hvO6uyloDlqs0Hdpjs?purpose=inline)

![Inside Open WebUI: How Browser Workers Bring Python, Plots, and Speech to Chat | by alex buzunov | Medium](https://images.openai.com/static-rsc-4/Gf9yzvF3qaKOsqi8vAdo3k3_Qpzn7O9_xt3jsdmYTW3eovgsVVo9pyyFByI10ulT-bt9JKjnDhsxh7ejeVBvFg8KmKY5qQ_nXZiNbjCptZG8YbCl5EMJsjBwIE2VN5mYkgiqPYKvBg_Cd6jGVZlGEGLVefdGNqefQGtItk8CArA?purpose=inline)

This architecture is scalable, testable, and follows Clean Architecture principles.

### 33. What I would say in a Senior AI Engineer interview (2-minute answer)

> "I implement global exception handling by creating custom business exceptions and registering centralized exception handlers using FastAPI's `@app.exception_handler`. This ensures every API returns a consistent error format with fields like `success`, `error_code`, `message`, `request_id`, and `timestamp`. I also log exceptions with structured logging and avoid exposing internal errors like SQL queries.
>
> For middleware, I use it for cross-cutting concerns such as request logging, request IDs, authentication, rate limiting, latency measurement, CORS, and compression. Middleware executes before and after every request, making it ideal for functionality that should apply across the entire application. In production AI applications like RAG systems, middleware is also useful for PII masking, token usage logging, and request tracing across multiple services."

### 34. Production Features Expected in Senior AI Projects

|
Feature

|

Used In

|
| --- | --- |
|

Global exception handlers

|

Every FastAPI project

|
|

Custom business exceptions

|

Service layer

|
|

Structured JSON errors

|

Enterprise APIs

|
|

Request ID middleware

|

Distributed systems

|
|

Logging middleware

|

Observability

|
|

Authentication middleware

|

JWT/OAuth

|
|

Rate limiting

|

Public APIs

|
|

PII masking

|

Enterprise AI

|
|

OpenTelemetry tracing

|

Microservices

|
|

Prometheus metrics

|

Production monitoring

|

This is the level of implementation and explanation typically expected in Senior FastAPI / Senior AI Engineer interviews (8+ years experience).



-----------------------







Use middleware for request/response concerns and exception handlers for turning errors into consistent HTTP responses.

### Request flow

Incoming request

Middleware wraps the entire request lifecycle.

Middleware

Logging, timing, correlation IDs, CORS, auth hooks.

Route → service → repository

Business logic raises exceptions when needed.

Exception handler

Maps errors to one JSON error format.

Response

Middleware can add headers before returning.

### Production pattern

Python

Run

from fastapi import FastAPI, HTTPException, Request

from fastapi.responses import JSONResponse

import time, uuid, logging

app = FastAPI()

log = logging.getLogger("api")

# ---------- Middleware ----------

@app.middleware("http")

async def request_context(request: Request, call_next):

request_id = str(uuid.uuid4())

start = time.perf_counter()

try:

response = await call_next(request)

except Exception:

log.exception("Unhandled error", extra={"request_id": request_id})

raise

response.headers["X-Request-ID"] = request_id

response.headers["X-Process-Time"] = (

f"{time.perf_counter() - start:.3f}s"

)

return response

# ---------- Global handlers ----------

@app.exception_handler(HTTPException)

async def http_error(_: Request, exc: HTTPException):

return JSONResponse(

status_code=exc.status_code,

content={"error": exc.detail}

)

@app.exception_handler(Exception)

async def unhandled(_: Request, exc: Exception):

return JSONResponse(

status_code=500,

content={"error": "Internal server error"}

)

# ---------- Route ----------

@app.get("/items/{item_id}")

async def get_item(item_id: int):

if item_id == 0:

raise HTTPException(404, "Item not found")

return {"id": item_id}

### Why this split?

|
Middleware

|

Exception handler

|
| --- | --- |
|

Runs for every request

|

Runs only on errors

|
|

Logging, timing, headers

|

Error → HTTP response

|
|

Can inspect both request and response

|

Keeps error format consistent

|

A clean architecture lets services raise domain exceptions, while the API layer converts them into HTTP responses in one place.
