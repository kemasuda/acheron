import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import init_to_value
import dill

def define_bins(min, max, d):
    bins = jnp.arange(min, max+1e-10, d)
    Nbin = len(bins) - 1
    bins_center = 0.5 * (bins[1:] + bins[:-1])
    return bins, bins_center, Nbin

from tinygp import kernels, GaussianProcess, transforms
class TwodHierarchical():
    def __init__(self, samples, xmin, xmax, dx, ymin, ymax, dy, bin_log, sample_log, xvals=None, yvals=None, vallabel=None):
        self.postsamples = samples
        if bin_log:
            self.postsamples[:,:,1] = np.log10(self.postsamples[:,:,1] * 1e9)
            age_upper = lambda mass: np.log10((-30 * (mass-1.25) + 5.) * 1e9)
        else:
            age_upper = lambda mass: -30 * (mass-1.25) + 5.

        self.xmin, self.xmax, self.dx, self.ymin, self.ymax, self.dy, self.bin_log, self.sample_log, self.xvals, self.yvals, self.vallabel = xmin, xmax, dx, ymin, ymax, dy, bin_log, sample_log, xvals, yvals, vallabel

        self.xbins, self.xbins_center, self.Nxbin = define_bins(xmin, xmax, dx)
        self.ybins, self.ybins_center, self.Nybin = define_bins(ymin, ymax, dy)

        self.ds = dx * dy
        self.Nbin = self.Nxbin * self.Nybin
        self.Xbins_center = jnp.tile(self.xbins_center, self.Nybin)
        self.Ybins_center = jnp.repeat(self.ybins_center, self.Nxbin)

        xidx = jnp.digitize(self.postsamples[:,:,0], self.xbins) - 1
        yidx = jnp.digitize(self.postsamples[:,:,1], self.ybins) - 1
        xyidx = xidx + yidx * self.Nxbin
        _xyidx = jnp.where((0<=xidx)&(xidx<self.Nxbin)&(0<=yidx)&(yidx<self.Nybin), xyidx, -1)

        if sample_log == bin_log:
            self.L = jnp.array([jnp.sum(_xyidx == k, axis=1) for k in range(self.Nbin)]).T
        elif sample_log: # log sampled but linear bin
            ageprior_correction = self.postsamples[:,:,1]
            self.L = jnp.array([jnp.sum((_xyidx == k)*ageprior_correction, axis=1) for k in range(self.Nbin)]).T
        else: # lin sampled but log bin
            ageprior_correction = jnp.exp(-self.postsamples[:,:,1]) # ??
            self.L = jnp.array([jnp.sum((_xyidx == k)*ageprior_correction, axis=1) for k in range(self.Nbin)]).T

        self.idx_valid_mass_age = (self.Ybins_center < age_upper(self.Xbins_center))
        self.idx_valid_grid = self.idx_valid_mass_age.reshape((self.Nybin,self.Nxbin))
        self.idx_valid_numgrid = jnp.where(self.idx_valid_grid, 1, jnp.nan)
        self.idx_valid_xdiff = jnp.diff(self.idx_valid_numgrid, axis=1)==0
        self.idx_valid_ydiff = jnp.diff(self.idx_valid_numgrid, axis=0)==0
        self.Nvalbin = jnp.sum(self.idx_valid_mass_age)
        self.eye = jnp.eye(self.Nvalbin)

        self.alpha_max = jnp.log(1./self.ds)
        #alpha_mean = jnp.log(1./(xmax-xmin)/(ymax-ymin))
        self.alpha_mean = jnp.log(1./dx/dy/np.sum(self.idx_valid_mass_age))

    def stepmodel(self, rflag=None, normprec=1e4, alpha_zero=-10, nodata=False, **kwargs):
        # hyperprior
        alphas = numpyro.sample("alphas", dist.Uniform(low=alpha_zero*jnp.ones(self.Nbin), high=self.alpha_max*jnp.ones(self.Nbin)))
        alphas = jnp.where(self.idx_valid_mass_age, alphas, alpha_zero)
        priors = jnp.exp(alphas)
        numpyro.deterministic("priors", priors)

        # normalization
        prob_sum = jnp.sum(priors*self.ds)
        norm_factor = -normprec * (1. - prob_sum)**2
        numpyro.deterministic("prob_sum", prob_sum)
        numpyro.factor("norm_factor", norm_factor)

        # regularization (smoothness prior)
        alphas_grid = alphas.reshape((self.Nybin, self.Nxbin))
        xdiff = jnp.diff(alphas_grid, axis=1)[self.idx_valid_xdiff]
        ydiff = jnp.diff(alphas_grid, axis=0)[self.idx_valid_ydiff]
        lneps = numpyro.sample("lneps", dist.Uniform(low=-5, high=5))
        log_reg = -0.5 * jnp.exp(lneps) * (jnp.sum(xdiff**2) + jnp.sum(ydiff**2)) + 0.5 * (self.Nvalbin-1) * lneps

        # add rotation data
        if rflag is not None:
            fracs = numpyro.sample("fracs", dist.Uniform(low=jnp.zeros(self.Nbin), high=jnp.ones(self.Nbin)))
            fracs = jnp.where(self.idx_valid_mass_age, fracs, 0)
            F = fracs**rflag[:,None] * (1.-fracs)**(1.-rflag[:,None])
        else:
            F = 1

        if nodata:
            log_marg_like = 0
        else:
            log_marg_like = jnp.sum(jnp.log(jnp.dot(F*self.L, priors*self.ds)))
        numpyro.factor("loglike", log_marg_like + log_reg)

    def gpmodel(self, rflag=None, normprec=1e4, alpha_zero=-10, float_mean=False, nodata=False,
        gpkernel='rbf', xmin=-4, xmax=1, ymin=-1, ymax=3, amin=-3, amax=3, smin=-5, smax=0, nojitter=True):
        # hyperprior
        alphas = numpyro.sample("alphas", dist.Uniform(low=alpha_zero*jnp.ones(self.Nbin), high=self.alpha_max*jnp.ones(self.Nbin)))
        alphas = jnp.where(self.idx_valid_mass_age, alphas, alpha_zero)
        priors = jnp.exp(alphas)
        numpyro.deterministic("priors", priors)

        # normalization
        prob_sum = jnp.sum(priors*self.ds)
        norm_factor = -normprec * (1. - prob_sum)**2
        numpyro.deterministic("prob_sum", prob_sum)
        numpyro.factor("norm_factor", norm_factor)

        # GP
        lnlenx = numpyro.sample("lnlenx", dist.Uniform(low=xmin, high=xmax))
        lnleny = numpyro.sample("lnleny", dist.Uniform(low=ymin, high=ymax))
        lna = numpyro.sample("lna", dist.Uniform(low=amin, high=amax))
        lenx, leny = jnp.exp(lnlenx), jnp.exp(lnleny)
        if nojitter:
            lnsigma = -5
        else:
            lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=smin, high=smax))
        if float_mean:
            alpha_mu = numpyro.sample("alpha_mu", dist.Uniform(low=alpha_zero, high=self.alpha_max))
            mu = alpha_mu
        else:
            mu = self.alpha_mean

        if gpkernel == 'rbf':
            kernel = kernels.ExpSquared(jnp.array([lenx, leny]))
        elif gpkernel == 'exp':
            kernel = kernels.Exp(jnp.array([lenx, leny]))
        elif gpkerenl == 'rquad':
            al = numpyro.sample("al", dist.Normal(loc=0, scale=5))
            kernel = kernels.RationalQuadratic(alpha=al, scale=jnp.array([lenx, leny]))

        kernel = jnp.exp(2*lna) * kernel
        X = jnp.array([self.Xbins_center[self.idx_valid_mass_age], self.Ybins_center[self.idx_valid_mass_age]]).T
        gp = GaussianProcess(kernel, X, diag=jnp.exp(2*lnsigma), mean=mu)
        loglike_gp = gp.condition(alphas[self.idx_valid_mass_age])

        # add rotation data
        if rflag is not None:
            fracs = numpyro.sample("fracs", dist.Uniform(low=jnp.zeros(self.Nbin), high=jnp.ones(self.Nbin)))
            fracs = jnp.where(self.idx_valid_mass_age, fracs, 0)
            F = fracs**rflag[:,None] * (1.-fracs)**(1.-rflag[:,None])
        else:
            F = 1

        if nodata:
            log_marg_like = 0
        else:
            log_marg_like = jnp.sum(jnp.log(jnp.dot(F*self.L, priors*self.ds)))

        numpyro.factor("loglike", log_marg_like + loglike_gp)

    """
    def gpmodel(self, rflag=None, eps=2, normprec=1e4, alpha_zero=-10,  xmin=-4, xmax=1, ymin=-1, ymax=3, amin=-3, amax=3, smin=-5, smax=3, float_mean=False, nojitter=False, gpkernel='rbf', nodata=False):
        print ("mygp")
        # GP
        lnlenx = numpyro.sample("lnlenx", dist.Uniform(low=xmin, high=xmax))
        lnleny = numpyro.sample("lnleny", dist.Uniform(low=ymin, high=ymax))
        lna = numpyro.sample("lna", dist.Uniform(low=amin, high=amax))
        lenx, leny = jnp.exp(lnlenx), jnp.exp(lnleny)

        dx2 = jnp.power((self.Xbins_center[:, None] - self.Xbins_center[None, :]) / lenx, 2.0)
        dy2 = jnp.power((self.Ybins_center[:, None] - self.Ybins_center[None, :]) / leny, 2.0)
        kernel = jnp.exp(2*lna) * jnp.exp(-0.5*dx2-0.5*dy2)

        if nojitter:
            lnsigma = -5
        else:
            lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=smin, high=smax))
        kernel += jnp.exp(2*lnsigma) * jnp.eye(self.Nbin)

        if float_mean:
            alpha_mu = numpyro.sample("alpha_mu", dist.Uniform(low=alpha_zero, high=self.alpha_max))
            mu = alpha_mu * jnp.ones(self.Nbin)
        else:
            mu = self.alpha_mean * jnp.ones(self.Nbin)

        mv = dist.MultivariateNormal(loc=mu, covariance_matrix=kernel)
        alphas = numpyro.sample("alphas", mv)
        alphas = jnp.where(self.idx_valid_mass_age, alphas, alpha_zero)
        alphas = jnp.where(alphas > alpha_zero, alphas, alpha_zero)
        priors = jnp.exp(alphas)
        numpyro.deterministic("priors", priors)

        # normalization
        prob_sum = jnp.sum(priors*self.ds)
        norm_factor = -normprec * (1. - prob_sum)**2
        numpyro.deterministic("prob_sum", prob_sum)
        numpyro.factor("norm_factor", norm_factor)

        # add rotation data
        if rflag is not None:
            fracs = numpyro.sample("fracs", dist.Uniform(low=jnp.zeros(self.Nbin), high=jnp.ones(self.Nbin)))
            fracs = jnp.where(self.idx_valid_mass_age, fracs, 0)
            F = fracs**rflag[:,None] * (1.-fracs)**(1.-rflag[:,None])
        else:
            F = 1

        log_marg_like = jnp.sum(jnp.log(jnp.dot(F*self.L, priors*self.ds)))
        numpyro.factor("loglike", log_marg_like)
    """

    def setup_hmc(self, target_accept_prob=0.95, num_warmup=1000, num_samples=1000, model='step'):
        self.n_sample = num_samples
        init_strategy = init_to_value(values={"alphas": self.alpha_mean*jnp.ones(self.Nbin)})
        if model == 'step':
            print ("# step model used.")
            kernel = numpyro.infer.NUTS(self.stepmodel, target_accept_prob=target_accept_prob, init_strategy=init_strategy)
        elif model == 'gp':
            print ("# gp model used.")
            kernel = numpyro.infer.NUTS(self.gpmodel, target_accept_prob=target_accept_prob, init_strategy=init_strategy)
        else:
            print ("# model should be either step or gp.")
        mcmc = numpyro.infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        self.mcmc = mcmc

    def run_hmc(self, rng_key, save=None, **kwargs):
        self.mcmc.run(rng_key, **kwargs)
        self.mcmc.print_summary()
        self.samples = self.mcmc.get_samples()
        if save is not None:
            with open(save+"_mcmc.pkl", "wb") as f:
                dill.dump(self.mcmc, f)

    def load_mcmc(self, filepath):
        self.mcmc = dill.load(open(filepath, 'rb'))
        self.samples = self.mcmc.get_samples()

    def summary_plots(self, xlabel, ylabel, rotflag=None, show_vals=False, save=None):
        samples = self.samples
        xbins, ybins = self.xbins, self.ybins
        Nxbin, Nybin = self.Nxbin, self.Nybin
        dx, dy = self.dx, self.dy
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        xvals, yvals = self.xvals, self.yvals
        n_sample = self.n_sample

        priors = samples['priors']
        pmean, pstd = jnp.mean(priors, axis=0), jnp.std(priors, axis=0)
        pmean, pstd = np.array(pmean.reshape((Nybin, Nxbin))), np.array(pstd.reshape((Nybin, Nxbin)))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(14,7))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.plot(dsim.mass_true, dsim.age_true, '.', color='pink', alpha=0.8, markersize=1)
        plt.imshow(pmean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto', origin='lower',  cmap=plt.cm.binary, alpha=0.8)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.colorbar(pad=0.02, label='probability density')
        plt.title("mean prediction")
        if save is not None:
            plt.savefig(save+"_2d.png", dpi=200, bbox_inches="tight")

        priors_grid = priors.reshape((n_sample, Nybin, Nxbin))
        pxs = jnp.sum(priors_grid, axis=1)*dy
        pys = jnp.sum(priors_grid, axis=2)*dx
        pxmean, pxstd = jnp.mean(pxs, axis=0), jnp.std(pxs, axis=0)
        pymean, pystd = jnp.mean(pys, axis=0), jnp.std(pys, axis=0)
        px95, py95 = jnp.percentile(pxs, 95, axis=0), jnp.percentile(pys, 95, axis=0)
        px5, py5 = jnp.percentile(pxs, 5, axis=0), jnp.percentile(pys, 5, axis=0)

        for bins, marg, margstd, label, med, (min, max), axis, p95, p5 in zip([xbins, ybins], [pxmean, pymean], [pxstd, pystd], [xlabel, ylabel], [xvals, yvals], [(xmin, xmax), (ymin, ymax)], ['x', 'y'], [px95, py95], [px5, py5]):
            plt.figure(figsize=(12,6))
            plt.xlabel(label)
            plt.ylabel("probability density")
            plt.xlim(min, max)
            if med is not None and show_vals:
                plt.hist(med, density=True, bins=bins, histtype='step', lw=1, ls='dashed', label=self.vallabel)
            plt.plot(np.repeat(bins, 2), np.r_[[0], np.repeat(marg, 2), [0]], color='C1', label='prediction (mean \& SD)')
            plt.fill_between(np.repeat(bins, 2), np.r_[[0], np.repeat(marg-margstd, 2), [0]], np.r_[[0], np.repeat(marg+margstd, 2), [0]], color='C1', alpha=0.2)
            #plt.fill_between(np.repeat(bins, 2), np.r_[[0], np.repeat(p5, 2), [0]], np.r_[[0], np.repeat(p95, 2), [0]], color='C1', alpha=0.1)
            plt.legend(loc='best')
            if save is not None:
                plt.savefig(save+"_%s.png"%axis, dpi=200, bbox_inches="tight")

        self.priors_grid = priors_grid

        if 'fracs' not in samples.keys():
            return None

        rotfracs = np.array(samples['fracs'])
        rotfracs[:,~self.idx_valid_mass_age] = np.nan
        rmean, rstd = jnp.mean(rotfracs, axis=0), jnp.std(rotfracs, axis=0)
        rmean = np.array(rmean.reshape((Nybin, Nxbin)))
        rstd = np.array(rstd.reshape((Nybin, Nxbin)))

        plt.figure(figsize=(14,7))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.plot(dsim.mass_true, dsim.age_true, '.', color='pink', alpha=0.8, markersize=1)
        plt.imshow(rmean, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto', origin='lower', cmap=plt.cm.binary, alpha=0.8, vmin=0, vmax=1)
        plt.plot(xvals, yvals, 'o', mfc='none', color='C0')
        if rotflag is not None:
            plt.plot(xvals[rotflag], yvals[rotflag], 'o', color='C0')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.colorbar(pad=0.02, label='probability density')
        plt.title("mean prediction")
        if save is not None:
            plt.savefig(save+"_fracs.png", dpi=200, bbox_inches="tight")
