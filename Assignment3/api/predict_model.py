import array
import numpy as np
import matplotlib.pyplot as plt
import joblib
from fastapi.responses import FileResponse
from numpy import array
from pydantic import BaseModel

from typing import Union
from datetime import datetime, timedelta
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext 
import sys


print(sys.path)
# sys.path.insert(1, 'scripts/')
from aux_functions import load_n_combine_df
(X_train,y_train),(X_validate,y_validate),(X_test,y_test) = load_n_combine_df(path_to_data='dataset/',features_to_keep=np.arange(0,1,1),class_labels=False)

loaded_model = joblib.load(open('modelLinearRegression.pkl', 'rb'))


def pred(X_validate,y_validate):

    yhat = loaded_model.predict(X_validate)
    mae = np.mean(np.abs(y_validate-yhat))
    rmse = np.sqrt(np.mean((y_validate-yhat)**2))
    bias = np.mean(y_validate-yhat)
    r2 = 1 - (np.sum((y_validate-yhat)**2))/(np.sum((y_validate-np.mean(y_validate))**2))
    out= 'MAE:{} flashes, RMSE:{} flashes, Bias:{} flashes, Rsquared:{}'.format(np.round(mae,2),np.round(rmse,2),np.round(bias,2),np.round(r2,2))

    return out

# ______________________________________________________

# JWT Authentication

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


fake_users_db = {
    "moksha": {
        "username": "moksha",
        "full_name": "Moksha Doshi",
        "email": "md@gmail.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    },
    "vyshnavi": {
        "username": "vyshnavi",
        "full_name": "Vyshnavi Pendru",
        "email": "vp@gmail.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": True,
    },
}



class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None

class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


app = FastAPI()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user



async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return {"item_id": "Foo", "owner": current_user.username}


@app.get("/users/me/scores/")
def prediction(current_user: User = Depends(get_current_active_user)):
    accuracy =pred(X_validate,y_validate)
    return {"Predction Scores:" : accuracy}

@app.get("/users/test/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


#get predictions 
@app.get("/users/me/predicting/", response_model=User)
def model_predict(current_user: User = Depends(get_current_active_user)):
    yhat = loaded_model.predict(X_validate)

#make figure  
    fig = plt.figure(figsize=(5,5))
    fig.set_facecolor('w')
    ax = plt.gca()

    ax.scatter(yhat,y_validate,s=1,marker='+')
    ax.plot([0,3500],[0,3500],'-k')
    ax.set_ylabel('ML Prediction, [$number of flashes$]')
    ax.set_xlabel('GLM measurement, [$number of flashes$]')

    fig.savefig('slr1.png')
    return FileResponse('slr1.png')

   
@app.get("/users/me/predict/{input}")
def model_predict(input:float,current_user: User = Depends(get_current_active_user)):
    y_values= array([input]).reshape(-1,1)
   
    y_pred = loaded_model.predict(y_values)
   
    return {"predictions":list(y_pred)}

