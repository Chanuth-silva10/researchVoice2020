# 2023-010

## Projects Installation Setup

<div class="termy">

Go to backend folder 

```console
$ cd backend
```

firstly we should create python envirnmoent

```console
$ python -m virtualenv env
```

for activate envirnment write this code(please remember now you still inside the backend folder)

```console
$ cd env/Scripts

$ activate
```

write this code for input the dependecncie set(please remember you should caom back backend folder using cd ../..) 

```console
$ pip install -r requirments.txt
```

After successfully dependencies we can run app using this code(now you are still inside the backend folder)

```console
$ uvicorn app.app:app --reload
```

### For connect API docs

Now go to <a href="http://127.0.0.1:8000/docs" class="external-link" target="_blank">http://127.0.0.1:8000/docs</a>.

