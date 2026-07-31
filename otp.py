import pyotp

S = ""
totp = pyotp.TOTP(S)
print("当前6位验证码：", totp.now())
