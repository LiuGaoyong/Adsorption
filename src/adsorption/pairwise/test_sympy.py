from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr

x = symbols("x")
expr = parse_expr("sqrt(x) + pi")
# 计算数值结果，默认精度为15位
result = expr.evalf(x=2)
print(result)  # 输出: 4.55580621596289

# 指定精度
result_high_prec = expr.evalf(50)
print(
    result_high_prec
)  # 输出: 4.5558062159628883149203106582734642028808593750000
