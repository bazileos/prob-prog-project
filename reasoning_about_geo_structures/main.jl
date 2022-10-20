using PyCall

py"""
def add20(x):
    return x + 20
"""

a= py"add20"(20)

print(a)    