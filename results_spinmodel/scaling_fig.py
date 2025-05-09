import os, argparse, time, gc, multiprocessing, pickle, psutil
import numpy as np

R2_LABELS = {'N1':r'$\hat{\theta}$',
'Nhld':r'$\theta^*_{hld}$' ,
'Nhld2':r'$\theta^*_{hld2}$' ,
}


# LABELS = {'Emp'        : 'Empirical', 
#           'N1'         : 'Newton 1step', 
#           'TUR'        : 'MTUR', 
#           'N h'     : 'Newton Holdout', 
#           'N h f'   : 'Newton HoldoutF', 
#           'N c h'     : 'Newton TRC Hold', 
#           'N c h f'   : 'Newton TRC HoldF', 
#           'T h'     : 'TRON Holdout',
#           'T h f'   : 'TRON HoldoutF',
#           'T na h'  : 'TRON Holdout NA',
#           'T na h f': 'TRON HoldoutF NA',
#           'G'       : 'Grad Ascent',
#           'G h'    : 'Grad Holdout',
#           'G h f'    : 'Grad Holdout F',
#           'BFGS'       : 'BFGS'}



if __name__ == "__main__":
    process = psutil.Process(os.getpid())

    # -------------------------------
    # Argument Parsing
    # -------------------------------
    parser = argparse.ArgumentParser(description="Estimate EP for the spin model with varying beta values.")

    # parser.add_argument("--num_steps", type=int, default=2**7,
    #                     help="Number of simulation steps (default: 128)")
    # parser.add_argument("--rep", type=int, default=1_000_000,
    #                     help="Number of repetitions for the simulation (default: 1,000,000)")
    # parser.add_argument("--size", type=int, default=100,
    #                     help="System size (default: 100)")
    parser.add_argument("--BASE_DIR", type=str, default="~/MaxEntData",
                        help="Base directory to store simulation results (default: '~/MaxEntData').")
    # parser.add_argument("--beta_min", type=float, default=0,
    #                     help="Minimum beta value (default: 0)")
    # parser.add_argument("--beta_max", type=float, default=4,
    #                     help="Maximum beta value (default: 4)")
    # parser.add_argument("--num_beta", type=int, default=101,
    #                     help="Number of beta values to simulate (default: 101)")
    # parser.add_argument("--J0", type=float, default=1.0,
    #                     help="Mean interaction coupling (default: 1.0)")
    # parser.add_argument("--DJ", type=float, default=0.5,
    #                     help="Variance of the quenched disorder (default: 0.5)")
    parser.add_argument('--noplot', action='store_true', default=False,
                         help='Disable plotting if specified')
    parser.add_argument("--max_spins", type=int, default=None,
                        help="Number of spins to caculate (default: all))")
    # parser.add_argument("--patterns", type=int, default=None,
    #                     help="Hopfield pattern density (default: None).")
    parser.add_argument("--overwrite", action="store_true",  default=False, help="Overwrite existing files.")
    args = parser.parse_known_args()[0]

    #N = args.size
    #rep = args.rep

    # -------------------------------
    # Global Setup
    # -------------------------------
    BASE_DIR = os.path.expanduser(args.BASE_DIR) + '/sequential/'
    DTYPE = 'float32'
    
    # -------------------------------
    # Run Experiments Across Beta Values
    # -------------------------------
    EPvals = {}
    betas = []

    # results = []
    N=None
    rep=None

    for file_name in sorted(os.listdir(BASE_DIR))[::-1]:
        start_time = time.time()

        if not file_name.endswith('.npz') or file_name == 'plot_data.npz':
            continue

        res = None

        out_filename = BASE_DIR+'epdata_' + os.path.basename(file_name)+'.pkl'
        if os.path.exists(out_filename):
            print(f'{out_filename} exists! '+('loading' if not args.overwrite else 'overwriting'))
            if not args.overwrite:
                with open(out_filename, 'rb') as file:
                    res = pickle.load(file)
                    
        if res is None:
            import calc
            f, f_kwargs = calc.calc, dict(file_name = BASE_DIR+file_name, max_spins=args.max_spins,)
            if True:
                with multiprocessing.Pool(processes=1) as pool:
                    res = pool.apply(f, kwds=f_kwargs)
            else:
                res = f(**f_kwargs)

            with open(out_filename, 'wb') as file:
                pickle.dump(res, file)
                print(f'Saved to {out_filename}')

        N = res['J'].shape[0]

        memory_usage = process.memory_info().rss / 1024 / 1024

        print(f"\n[Results] {time.time()-start_time:3f}s  mem={memory_usage:.1f}mb")
        for k in res['ep']:
            print(f"  EP ({k:15s})    : {res['ep'][k]:.6f}   {res['times'].get(k,np.nan):3f}s")
        print("-" * 70)

        # res = calc.calc(BASE_DIR+file_name, 
        #     overwrite=args.overwrite, 
        #     grad=not args.nograd, 
        #     newton=not args.nonewton)


        beta = res['beta']
        for k, v in res['ep'].items():
            #if k.endswith(' tst'):
            #    continue
            if k not in EPvals:
                EPvals[k] = []
            EPvals[k].append( (beta, v) )

        J      = res['J']
        xvals = (J - J.T)[:]
        yvals = {}
        R2    = {}
        r2methods = ['N1',#'Nhld',
            'Nhld2'] #,'NS']
        for k in r2methods:
            if k not in res['thetas']:
                continue
            Thetas = np.vstack([np.concatenate([m[:i], [0,], m[i:]]) 
                                for i,m in enumerate(res['thetas'][k])])
            yy = (Thetas - Thetas.T)[:]

            R2[k] = 1- np.mean( (yy-xvals)**2 ) / np.mean( (yy-yy.mean())**2 )
            #print(np.mean( (yy-xvals)**2 ),np.mean( (yy-yy.mean())**2 ), yy.mean())
            #print(np.isnan(yy).any())
            yvals[k]=yy

        print(f'theta R2 values: ' + 
            " ".join([k+f'={v:3f}' for k,v in R2.items()]) )

        if False and 4>beta >= 1.9:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(8,4), layout='constrained')
            m1a,m2a,m1b,m2b=0,0,0,0
            for k in r2methods:
                cyy = yvals[k]
                m1a = max(m1a,-cyy.min())
                m2a = max(m2a, cyy.max())
                plt.subplot(1,2,1)
                plt.scatter(xvals, cyy, alpha=0.3, 
                    label=R2_LABELS[k]+r'$\quad(R^2='+f'{R2[k]:3.3f}'+')$', s=5, edgecolor='none')
                plt.subplot(1,2,2)
                cyy[np.abs(xvals)>2]=np.nan
                # xvals[np.isclose(xvals,0)]=np.nan
                m1b = max(m1b,-np.nanmin(cyy))
                m2b = max(m2b, np.nanmax(cyy))
                plt.scatter(xvals, cyy, alpha=0.3, 
                    label=R2_LABELS[k]+r'$\quad(R^2='+f'{R2[k]:3.3f}'+')$', s=5, edgecolor='none')
            for sbi in [1,2]:
                plt.subplot(1,2,sbi)
                m1,m2=(m1a,m2a) if sbi==1 else (m1b,m2b)
                lims = [-1.1*m1, 1.1*m2]
                print(lims)
                plt.xlim( *lims )
                plt.ylim( *lims )
                plt.plot(lims, lims, lw=1, ls=":", c='k')

            plt.legend()
            plt.xlabel(r'$\beta(w_{ij}-w_{ji})$')
            plt.ylabel(r'$\theta_{ij}-\theta_{ji}$')
            plt.title(fr'$\beta={beta:3.3f}$')
            plt.show()

            asdf

        del xvals, yvals, R2, res, J
        gc.collect()


    if not args.noplot:
        from matplotlib import pyplot as plt

        #EP    = np.array(EPvals).T
        #betas = np.array(betas)

        # # -------------------------------
        # # Save results
        # # -------------------------------
        # SAVE_DATA_DIR = BASE_DIR
        # if not os.path.exists(SAVE_DATA_DIR):
        #     print(f'Creating base directory: {SAVE_DATA_DIR}')
        #     os.makedirs(SAVE_DATA_DIR)
        # filename = f"{SAVE_DATA_DIR}plot_data.npz"
        # print(filename)
        # np.savez(filename, EP=EP, betas=betas)
        # print(f'Saved calculations to {filename}')
        # -------------------------------
        # Plot Results
        # -------------------------------
        plt.rc('text', usetex=True)
        plt.rc('font', size=22, family='serif', serif=['latin modern roman'])
        plt.rc('legend', fontsize=20)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')


        plt.figure(figsize=(5, 5), layout='constrained')
        cmap = plt.get_cmap('inferno_r')
        b1,b2 = 0,0
        mxep  = 0

        LEGEND_LABELS = {
            'Emp':r'$\Sigma$', 
            'TUR':r'$\Sigma_{\bm g}^\textnormal{\small TUR}$', 
            'N1':r'$\widehat{\Sigma}_{\bm g}$', 
        }
        for k in EPvals:
            if k not in LEGEND_LABELS:
                LEGEND_LABELS[k] = r'${\Sigma}_{\bm g}^\text{'+k+'}$'

        for k, v in EPvals.items():
            betas, eps = map(np.array, zip(*v))
            s_ixs = np.argsort(betas)
            plt.plot(betas[s_ixs], eps[s_ixs], label=LEGEND_LABELS[k], lw=2) # color=colors[i-1], lw=2)
            b1 = min(b1, betas.min())
            b2 = max(b2, betas.max())
            if eps.max() < 10:
                mxep = max(mxep, eps.max())

        plt.axis([b1, b2, 0, mxep * 1.05])
        plt.ylabel(r'$\Sigma$', rotation=0, labelpad=20)
        plt.xlabel(r'$\beta$')
        plt.title(f'N={N}')

        # Legend
        plt.legend(
            ncol=1,
            columnspacing=0.5,
            handlelength=1.0,
            handletextpad=0.5,
            loc='best'
        )

        # Save and show figure

        if not os.path.exists('img'):
            os.makedirs('img')
        plt.savefig('img/Fig_1a.pdf', bbox_inches='tight')
        plt.show()

