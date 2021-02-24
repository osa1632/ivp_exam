import numpy as np
import  matplotlib.pyplot as plt
import argparse
import sys
import warnings
import tqdm

''''
u_t = i [ [0 1]   u_xx + [3  -1   u ]   0<=x<=1  , t>=0         u(0, x) = [    cos w x
           1 0]]          -1  3]                                            -i sin w x    ]
'''
i = 1j


def simulate(args):
    # warnings.filterwarnings('error')

    N, method = args.N, args.method_map[args.method]

    h = 1/N
    x = np.arange(0, 1, h)
    k = (args.gamma)*h**(args.p)
    omega = 2*np.pi

    u = np.stack([np.cos(omega*x),
                        -i*np.sin(omega*x)],1)

    C_2x2 = i*np.array([[3,-1],[-1,3]])
    A_2x2 = i*np.array([[0,1],[1,0]])
    I_2x2 = np.eye(2)

    I_nxn = np.eye(N)
    O_nxn = np.zeros(N)
    F_nxn = np.roll(I_nxn,1,0)
    B_nxn = np.roll(I_nxn,-1,0)

    I = np.kron(I_2x2,I_nxn)
    O = np.kron(I_2x2, O_nxn)
    F = np.kron(A_2x2, F_nxn)
    B = np.kron(A_2x2, B_nxn)
    C = np.kron(C_2x2, I_nxn)

    u = u.reshape(-1)
    t0 = 0


    if method in ['LF','DF']:
        u = np.concatenate([(I + k * C - 2 * i * k / h ** 2 * A + (i * A * k / h ** 2).dot(F + B)).dot(u),u])
        t0+=k

    ts = np.arange(t0,args.tf+k,k)

    for t in ts:
        if method=='FE':
            u = (I+k/h**2*(F-2*I+B)+k*C).dot(u)
        elif method=='BE':
            u = np.linalg.inv(I+k/h**2*(F-2*I+B)).dot(I+k*C).dot(u)
            # u = np.linalg.inv(2*i*k/h**2*A-(i*A*k/h**2)).dot((F+B)).dot(I+k*C ).dot(u)
        elif method=='LF':
            raise NotImplementedError(method)
        elif method=='CN':
            u = np.linalg.inv(I+k/h**2/2*(F-2*I+B)).dot(I+k/h**2/2*(F-2*I+B)+k*C).dot(u)

            # u = np.linalg.inv((2*i*k/h**2*A-(i*A*k/h**2))/2).dot((F+B)).dot(I+k*C+(2*i*k/h**2*A-(i*A*k/h**2))/2 ).dot(u)
        elif method == 'DF':
            u = np.concatenate([np.concatenate([np.linalg.inv(I+i*k/h**2*A), O],0),np.concatenate([O,I],0)],1).dot(
             np.concatenate([np.concatenate([i*k/h**2*A*(B+F), -i*k/h**2*A],0),np.concatenate([I,O],0)],1)
            ).dot(u)
        else:
            pass
            # u = np.linalg.inv((2*i*k/h**2*A-(i*A*k/h**2))/2).dot((F+B)).dot(I+k*C+(2*i*k/h**2*A-(i*A*k/h**2))/2 ).dot(u)

    u = u.reshape((-1,2))
    if method in ['LF','DF']:
        u = u[:len(u)//2,:]
    plot(args, u, x)
    warnings.filterwarnings('default')

    return np.linalg.norm(np.abs(u)/np.sqrt(N))


def plot(args, u, x):
    if args.show:
        f = plt.figure(2)
        plt.plot(x, np.real(u),'-')
        plt.plot(x, np.imag(u),'-.')
        plt.ion()
        plt.show()
        plt.pause(5)
        # plt.close(f)
        plt.figure(1)


def build_args(methods, possible_Ns):
    ap.add_argument('--method', type=str, default='Forward-Euler',
                    choices=methods)
    ap.add_argument('--N', type=int, default=128  , choices=possible_Ns)
    ap.add_argument('--p', type=int, default=1)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--tf', type=float, default=1.)
    ap.add_argument('--show', action='store_true',default=1)

    aconyms = lambda s:''.join(w[0] for w in s.split('-'))

    args = ap.parse_args()

    method_map = {aconyms(m):m for m in methods}
    method_map.update({m:aconyms(m) for m in methods})
    method_map.update({aconyms(m):aconyms(m) for m in methods})
    setattr(args, 'method_map', method_map)
    return args

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="IVP")
    possible_Ns = [2**ii for ii in range(4, 6)]
    methods = ['Forward-Euler','Backward-Euler', 'Leaf-Frog','Crack-Nicholson','DuFort-Frenkel']
    args = build_args(methods, possible_Ns)
    # args.show=1
    methods=['CN']
    # possible_Ns=[16,32]
    for method in tqdm.tqdm(methods):
        errs = []
        args.method = method
        if True:#try:
            for N in possible_Ns:
                args.N = N
                errs.append(simulate(args))
        # except:
        #     continue
        xs, ys = np.log(possible_Ns), np.log(errs)
        z = np.polyfit(xs, ys, 1)
        plt.plot(xs, ys,'o-',label=method+f'({z[1]:0.5f})')

        # plt.text(np.nanmean(xs),np.nanmean(ys),'{:0}+{:1}*x'.format(z[1],z[0]))

        if True:  # np.isfinite(ys).all():# and abs(z[0]-2)<5:

            ret_val = '0'  # str(int(1000*z[0])/1000)
            plt.plot(xs, np.poly1d(z)(xs))
    plt.legend()
    plt.show(block=True)