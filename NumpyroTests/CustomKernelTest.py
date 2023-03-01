import matplotlib.pylab as plt
import jax
import jax.numpy as jnp
import numpyro
import tinygp

#===========================

class my_kernel(tinygp.kernels.Custom):
    '''
    Can't make a GP out of this
    '''
    def __init__(self,a,b,c):
        self.a=a
        self.b=b
        self.c=c

    def evaluate(self, X1, X2):
        if X1[1]==X2[1]==0:
            return(self.a)
        elif X1[1]!=X2[1]:
            return(self.b)
        else:
            return(self.c)

class my_kernel_nobools(tinygp.kernels.Custom):
    '''
    Can make a GP out of this
    '''
    def __init__(self,a,b,c):
        self.a=a
        self.b=b
        self.c=c

    def evaluate(self, X1, X2):
        out = 0
        out += self.a * abs((X1[1]-0)*(X2[1]-0))
        out += self.b * abs(X1[1]-X2[1])
        out += self.c * abs((X1[1] - 1) * (X2[1] - 1))

        return(out)


def smoothfunc(dx,b1,b2):
    if abs(dx)>b1+b2:
        return(1/jnp.cosh(dx*1+(1/b1+1/b2)**2))
    else:
        return(1/jnp.exp(abs(dx)*1+(1/b1+1/b2)**2))


class my_kernel_funcs(tinygp.kernels.Custom):
    '''
    Can make a GP out of this
    '''
    def __init__(self,lags, widths):
        self.lags   =   lags
        self.widths =   widths

    def evaluate(self, X1, X2):
        t1, t2 = X1[0], X2[0]
        a, b = X1[1], X2[1] #get bands

        dx = (t1-self.lags[a]) - (t2-self.lags[b])

        return(smoothfunc(dx,self.widths[a],self.widths[b]))



#===========================
mk      = my_kernel(a=0,b=1,    c=2)
mknb    = my_kernel_nobools(a=0,b=1,c=2)
mkf     = my_kernel_funcs(lags=jnp.array([0,1]),widths=jnp.array([1,2]))

X1 = [10,0]
X2 = [20,1]

def tform(A):
    I = len(A)
    J = len(A[0])

    Xs=jnp.array([A[i][0] for i in range(I)],dtype='float32')
    Is=jnp.array([A[i][1] for i in range(I)],dtype='int32')

    out=(Xs,Is)

    return(out)

#Test
print(mkf.evaluate(X1,X1))
print(mkf.evaluate(X1,X2))
print(mkf.evaluate(X2,X1))
print(mkf.evaluate(X2,X2))

data = tform([X1,X2])

gp=tinygp.GaussianProcess(
    kernel = mkf,
    X=jnp.array([X1,X2])
)

print("GP Made")
#===========================

A=gp.condition(jnp.array([1,2]),jnp.array([X1,X2])).gp.loc
print(A)