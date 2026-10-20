"""Stakeholder-side learning and proposal generation.

simulation.py calls ModelAgent.propose(Context). The agent extracts public
training data, transforms it, fits a surrogate, predicts unmeasured candidates,
and applies a proposal rule. It returns a table to submission_processing.py.

Surrogates see standardized inputs and standardized log targets. Their Prediction
arrays have one element per input row. ModelAgent converts these predictions to
original outcome units before sending them to the facility. Optional libraries
are imported only inside the model methods that need them.

MODELS and RULES at the bottom are extension registries. CSVSubmissionAgent is an
alternative interface adapter for historical files, not a retrainable model.
"""
import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import ShuffleSplit, LeaveOneOut, train_test_split
from config import Prediction

class GaussianProcessSurrogate:
    """Exact GP model wrapper with original GPyTorch or optional sklearn backend.
    """
    def __init__(self, seed=42, backend='gpytorch', training_iter=100, learning_rate=0.1):
        """Configure optimizer/backend and reproducible initialization for one GP.
        """
        self.seed, self.backend = seed, backend
        self.training_iter, self.learning_rate = training_iter, learning_rate

    def fit(self, X, y):
        """Fit a GP to X of shape (n_samples,n_features) and transformed y.
        
        GPyTorch uses a constant mean, scaled RBF kernel, Gaussian likelihood and
        marginal-likelihood training. The sklearn backend is a demo alternative.
        """
        # The lightweight demo backend avoids importing PyTorch/GPyTorch.
        if self.backend == 'sklearn':
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF
            self.model = GaussianProcessRegressor(kernel=ConstantKernel()*RBF(), alpha=1e-4, random_state=self.seed)
            self.model.fit(X, y)
            return
        if self.backend != 'gpytorch':
            raise ValueError('GP backend must be gpytorch or sklearn.')
        import torch
        import gpytorch
        # Isolate this model's initialization from other agents' torch RNG.
        # Restore global torch RNG afterward so one agent initialization does not perturb another.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            class ExactGP(gpytorch.models.ExactGP):
                """Local GPyTorch class connecting a mean function and covariance kernel.
                """
                def __init__(self, x, y, likelihood):
                    """Initialize the exact-inference training data, likelihood, mean and RBF kernel.
                    """
                    super().__init__(x, y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    # The kernel represents correlation between candidate inputs; ScaleKernel learns amplitude.
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                def forward(self, x):
                    """Build the multivariate Gaussian latent function distribution at input x.
                    """
                    return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))
            x, yy = torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
            self.model = ExactGP(x, yy, likelihood).double()
            self.model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.model)
            for _ in range(self.training_iter):
                optimizer.zero_grad()
                # Optimizers minimize: negate log marginal likelihood to maximize the GP evidence.
                loss = -mll(self.model(x), yy)
                loss.backward()
                optimizer.step()
            self.model.eval()
            likelihood.eval()

    def predict(self, X):
        """Return latent predictive mean/std for each transformed candidate row.
        
        Likelihood noise is not added to the GPyTorch uncertainty, matching the
        notebook use of model(x), rather than likelihood(model(x)).
        """
        # The lightweight demo backend avoids importing PyTorch/GPyTorch.
        if self.backend == 'sklearn':
            m, s = self.model.predict(X, return_std=True)
            return Prediction(m, s)
        import torch
        import gpytorch
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Query latent GP uncertainty, excluding likelihood observation noise.
            p = self.model(torch.as_tensor(X, dtype=torch.float64))
            return Prediction(p.mean.numpy(), p.variance.clamp_min(0).sqrt().numpy())

class BijectorPolynomialSurrogate:
    """Compare polynomial ensembles on x and reciprocal x, choosing lower spread.
    """
    def __init__(self, seed=42, n_splits=25, reciprocal_floor=1e-6):
        """Configure resampling and the guard used near zero reciprocal inputs.
        """
        self.seed, self.n_splits, self.floor = seed, n_splits, reciprocal_floor

    def reciprocal(self, X):
        """Compute 1/x while flooring tiny absolute inputs and preserving their sign.
        """
        safe = np.where(np.abs(X) < self.floor, np.where(X < 0, -self.floor, self.floor), X)
        return 1.0 / safe

    def fit(self, X, y):
        """Fit resampled polynomial interpolants in both input representations.
        
        Choose the largest total polynomial degree supportable by n-1 observations.
        Least squares handles singular designs instead of explicitly inverting them.
        """
        n, d = X.shape
        degree = 0
        # Count multivariate polynomial terms to avoid a degree unsupported by measured data.
        while comb(d + degree + 1, degree + 1, exact=True) <= n - 1:
            degree += 1
        needed = comb(d + degree, degree, exact=True)
        splitter = LeaveOneOut() if n-needed == 1 else ShuffleSplit(n_splits=self.n_splits, train_size=needed, random_state=self.seed)
        # Reuse identical training subsets for x and reciprocal x for a fair comparison.
        splits = [train for train, _ in splitter.split(X)]
        self.bases = []
        for transformed in (X, self.reciprocal(X)):
            poly = PolynomialFeatures(degree)
            design = poly.fit_transform(transformed)
            # lstsq also handles singular interpolation designs explicitly.
            # Solve the resampled polynomial systems without an unstable explicit matrix inverse.
            coefficients = [np.linalg.lstsq(design[idx], y[idx], rcond=None)[0] for idx in splits]
            self.bases.append((poly, np.asarray(coefficients)))

    def predict(self, X):
        """Predict with both ensembles; select lower-spread representation per row.
        
        Spread is ensemble population std, not the width between quantiles.
        """
        ensembles = [poly.transform(z) @ coefficients.T for z, (poly, coefficients) in zip((X, self.reciprocal(X)), self.bases)]
        means = np.stack([e.mean(axis=1) for e in ensembles])
        stds = np.stack([e.std(axis=1) for e in ensembles])
        # Each candidate can choose a different input representation.
        chosen = stds.argmin(axis=0)
        cols = np.arange(len(X))
        return Prediction(means[chosen, cols], stds[chosen, cols])

class BaggedXGBoostSurrogate:
    """Random-feature filtering followed by resampled XGBoost regressors.
    """
    def __init__(self, seed=42, n_models=None, max_depth=3, learning_rate=1.0, n_estimators=100):
        """Configure ensemble size and shared XGBoost hyperparameters.
        """
        self.seed, self.n_models = seed, n_models
        self.params = dict(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, n_jobs=1)

    def fit(self, X, y):
        """Filter features against a random dummy, then train the model ensemble.
        
        Fit uses only measured data. As in the notebook, an outer 80% subset precedes
        inner 80% resampling; unused outer rows are not validation metrics here.
        """
        from xgboost import XGBRegressor
        rng = np.random.default_rng(self.seed)
        # Add a random dummy feature as an importance threshold, not as a final predictor.
        augmented = np.column_stack([X, rng.normal(size=len(X))])
        selector = XGBRegressor(**self.params, random_state=self.seed).fit(augmented, y)
        # Keep original features that beat the random feature. If none pass, retain all.
        # Only original features compete against the dummy feature importance.
        self.mask = selector.feature_importances_[:-1] > selector.feature_importances_[-1]
        # Avoid a model with zero input columns when filtering removes everything.
        if not self.mask.any():
            self.mask[:] = True
        xx = X[:, self.mask]
        # Preserve the notebook outer training subset before inner ensemble resampling.
        train, _ = train_test_split(np.arange(len(X)), train_size=0.8, random_state=self.seed)
        splitter = ShuffleSplit(n_splits=self.n_models or len(X)-1, train_size=0.8, random_state=self.seed)
        self.models = []
        for i, (idx, _) in enumerate(splitter.split(train)):
            rows = train[idx]
            self.models.append(XGBRegressor(**self.params, random_state=self.seed+i).fit(xx[rows], y[rows]))

    def predict(self, X):
        """Return ensemble mean and population standard deviation per candidate.
        """
        # Rows are candidates; columns are independently fitted ensemble members.
        samples = np.column_stack([m.predict(X[:, self.mask]) for m in self.models])
        return Prediction(samples.mean(axis=1), samples.std(axis=1))

class ChebyshevBayesianSurrogate:
    """Chebyshev feature expansion with Bayesian ridge predictive uncertainty.
    """
    def __init__(self, seed=42, degree=10):
        """Choose how many Chebyshev terms to retain per input feature.
        """
        if degree < 1:
            raise ValueError('degree must be positive.')
        self.degree = degree

    def transform(self, X):
        # Preserve notebook convention: terms T0 through T(degree-1), per feature.
        """Expand each input dimension into T0,...,T(degree-1), then concatenate.
        """
        # Expand each dimension separately, preserving the notebook basis convention.
        return np.concatenate([np.polynomial.chebyshev.chebvander(X[:, j], self.degree-1) for j in range(X.shape[1])], axis=1)

    def fit(self, X, y):
        """Fit Bayesian ridge using expanded measured inputs and transformed targets.
        """
        self.model = BayesianRidge().fit(self.transform(X), y)

    def predict(self, X):
        """Return Bayesian ridge mean/std on expanded candidate features.
        """
        m, s = self.model.predict(self.transform(X), return_std=True)
        return Prediction(m, s)

class LorenzEnsembleSurrogate:
    """Independent neural networks containing a differentiable Lorenz layer.
    """
    def __init__(self, seed=42, n_models=5, epochs=9000, learning_rate=1e-3, steps=20):
        """Configure member count, optimization and Lorenz integration steps.
        """
        self.seed, self.n_models, self.epochs = seed, n_models, epochs
        self.learning_rate, self.steps = learning_rate, steps

    def fit(self, X, y):
        """Train full-batch neural members independently on public observations.
        
        A 3D hidden representation feeds an explicit-Euler Lorenz integration layer.
        The first linear layer adapts to the configured number of features.
        """
        import torch
        from torch import nn
        steps = self.steps
        class Lorenz(nn.Module):
            """Differentiable numerical Lorenz dynamics used as a hidden neural layer.
            """
            def forward(self, inputs):
                """Integrate three hidden coordinates together using Euler steps of dt=0.02.
                
                Compute all derivatives from the previous state before updating coordinates.
                """
                # Use each of the three hidden coordinates to initialize its own Lorenz state.
                x, yy, z = (inputs[:, j] + 1 for j in range(3))
                for _ in range(steps):
                    # Lorenz derivatives: sigma=10, rho=28, beta=8/3.
                    dx = 10*(yy-x)
                    dy = x*(28-z)-yy
                    dz = x*yy-(8/3)*z
                    # Simultaneous Euler update: derivatives above all use the previous state.
                    x, yy, z = x+0.02*dx, yy+0.02*dy, z+0.02*dz
                return torch.stack([x, yy, z], dim=1)/10000
        xx = torch.as_tensor(X, dtype=torch.float32)
        target = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.models = []
        # Restore global torch RNG afterward so one agent initialization does not perturb another.
        with torch.random.fork_rng(devices=[]):
            for i in range(self.n_models):
                torch.manual_seed(self.seed+i)
                # Learn a 3D representation, evolve it through Lorenz dynamics, then regress one target.
                model = nn.Sequential(nn.Linear(X.shape[1], 24), nn.SiLU(), nn.Linear(24, 3), nn.LayerNorm(3), Lorenz(), nn.LayerNorm(3), nn.Linear(3, 12), nn.GELU(), nn.Linear(12, 1))
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, epochs=self.epochs, steps_per_epoch=1)
                for _ in range(self.epochs):
                    optimizer.zero_grad()
                    # Full-batch mean squared error on standardized log targets.
                    loss = (model(xx)-target).square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite Lorenz training loss.')
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                self.models.append(model.eval())

    def predict(self, X):
        """Evaluate all trained members and return across-member mean/std.
        """
        import torch
        with torch.no_grad():
            # Rows are candidates; columns are independently fitted ensemble members.
            samples = np.column_stack([m(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel() for m in self.models])
        return Prediction(samples.mean(axis=1), samples.std(axis=1))

# Add a new model by registering a class with fit(X,y) and predict(X).
# Extension point: register new surrogate classes; the simulation loop stays unchanged.
MODELS = {'gp': GaussianProcessSurrogate, 'bijector': BijectorPolynomialSurrogate,
          'xgb': BaggedXGBoostSurrogate, 'chebyshev': ChebyshevBayesianSurrogate,
          'lorenz': LorenzEnsembleSurrogate}

def upper_score(prediction, observed_y, direction, beta, xi):
    """Assign optimistic acquisition values; larger values are always preferred.
    
    For default maximization use the stored log-normal upper bound when present.
    Otherwise use mean+beta*std, or -mean+beta*std for minimization.
    """
    if direction == 'maximize':
        return prediction.score_upper if prediction.score_upper is not None and beta == 1 else prediction.mean+beta*prediction.std
    return -prediction.mean+beta*prediction.std

def uncertainty_score(prediction, observed_y, direction, beta, xi):
    """Prefer candidates with larger original-scale predictive standard deviation.
    """
    return prediction.std

def ei_score(prediction, observed_y, direction, beta, xi):
    """Compute stakeholder EI using original-scale Gaussian moment approximation.
    
    The sign maps minimization to maximization. Near-zero uncertainty uses the
    deterministic positive-improvement limit.
    """
    sign = 1 if direction == 'maximize' else -1
    improvement = sign*prediction.mean - np.max(sign*np.asarray(observed_y)) - xi
    sigma = prediction.std
    z = improvement/np.maximum(sigma, 1e-12)
    return np.where(sigma > 1e-12, improvement*norm.cdf(z)+sigma*norm.pdf(z), np.maximum(improvement, 0))

# Extension point: score functions accept the same inputs regardless of model family.
RULES = {'upper': upper_score, 'uncertainty': uncertainty_score, 'ei': ei_score}

class ModelAgent:
    """Compose a surrogate and proposal rule into a stakeholder.
    
    Simulation supplies the complete current public history on each propose()
    call, so this implementation retrains there rather than needing observe().
    """
    def __init__(self, config):
        """Create this stakeholder own model instance from the MODELS registry.
        """
        self.config = config
        self.agent_id = config.agent_id
        # Construct a separate model object for this stakeholder, even if others use the same class.
        self.model = MODELS[config.model](seed=config.seed, **config.parameters)

    def propose(self, context):
        """Train on public data and return scored, ID-aligned candidate submissions.
        
        Pipeline: standardize X -> standardize log(y) -> fit surrogate -> predict ->
        convert log-normal moments to original units -> score -> optional shortlist.
        The returned uncertainty is a true std; score can use a different statistic.
        """
        # Training uses measured parameters only; unmeasured outcomes are absent from Context.
        X = context.observations[list(context.features)].to_numpy(float)
        y = context.observations[context.target].to_numpy(float)
        if len(y) < 4 or np.any(y <= 0):
            raise ValueError('Notebook-style agents require >=4 observations and strictly positive targets for log transform.')
        # Fit centering/scaling on public measured data, then reuse it for candidate features.
        scaler = StandardScaler().fit(X)
        # The original notebook uses log targets; strict positivity is checked above.
        log_y = np.log(y)
        center, scale = log_y.mean(), log_y.std()
        if scale < 1e-12:
            scale = 1.0
        # The surrogate sees standardized X and standardized log(y), not original units.
        self.model.fit(scaler.transform(X), (log_y-center)/scale)
        latent = self.model.predict(scaler.transform(context.candidates[list(context.features)].to_numpy(float)))
        # Moment match log-normal predictions to original-scale mean and std.
        # Undo log-target standardization while staying in log space.
        m, s = center + scale*latent.mean, scale*latent.std
        # Overflow remains nonfinite and is rejected by submission_processing; it is not hidden.
        with np.errstate(over='ignore', invalid='ignore'):
            # For log-normal Y: E[Y]=exp(m+s^2/2), whereas exp(m) is the median.
            mean = np.exp(m + s*s/2)
            # Use the log-normal variance formula; interval width would not be a valid std.
            std = np.sqrt(np.expm1(s*s)*np.exp(2*m+s*s))
            # Preserve the notebook optimistic one-log-std bound as a separate score statistic.
            upper = np.exp(m+s)
        prediction = Prediction(mean, std, upper)
        scores = RULES[self.config.rule](prediction, y, context.direction, self.config.beta, self.config.xi)
        # Attach predictions to the ORIGINAL candidate IDs in the same prediction row order.
        f = pd.DataFrame({'agent_id': self.agent_id, 'candidate_id': context.candidates.candidate_id.to_numpy(), 'prediction': mean, 'uncertainty': std, 'score': scores})
        # Sort scores without altering the candidate identity associated with each value.
        f = f.sort_values('score', ascending=False, kind='stable')
        return f.head(self.config.proposal_limit) if self.config.proposal_limit is not None else f

class CSVSubmissionAgent:
    """Adapt external proposal files to the same propose(Context) interface.
    
    CSV replay does not retrain when the shared measurement history changes.
    """
    def __init__(self, agent_id, path_template):
        """Store stakeholder identity and a file template containing optional {round}.
        """
        self.agent_id, self.path_template = agent_id, path_template
    def propose(self, context):
        """Load this round file and attach identity; processing validates it later.
        """
        f = pd.read_csv(self.path_template.format(round=context.round_id), dtype={'candidate_id': str, 'Index': str, 'index': str, 'pool_id': str})
        f['agent_id'] = self.agent_id
        return f


# WHOLE-AGENT REGISTRY: separate from MODELS and RULES above.
# Factories/classes accept (AgentConfig, *, base_directory='') and return an
# object with a unique agent_id and callable propose(Context) method.
AGENTS = {}

def register_agent(name):
    """Decorator for a whole-agent class or factory; duplicate names are errors.

    Example: @register_agent('my_llm') above your own LLM adapter class.
    The adapter itself supplies API integration; registration does not call an LLM.
    """
    def decorate(factory):
        if not isinstance(name, str) or not name or name in AGENTS:
            raise ValueError(f'Invalid or duplicate agent type: {name!r}')
        if not callable(factory):
            raise TypeError('An agent factory must be callable.')
        AGENTS[name] = factory
        return factory
    return decorate

@register_agent('model')
def model_agent_factory(config, *, base_directory=''):
    """Adapt existing ModelAgent without changing its model/proposal algorithms."""
    return ModelAgent(config)

@register_agent('csv')
def csv_agent_factory(config, *, base_directory=''):
    """Resolve replay-file paths here rather than adding CSV branches to main.py."""
    from pathlib import Path
    template = config.parameters.get('path_template')
    if not isinstance(template, str) or not template:
        raise ValueError('CSV agent needs parameters.path_template.')
    path = Path(template)
    if not path.is_absolute():
        path = Path(base_directory or '.')/path
    return CSVSubmissionAgent(config.agent_id, str(path))

def build_agent(config, *, base_directory=''):
    """Build any registered stakeholder through one interface.

    Explicit agent_type selects a whole-agent implementation. Without it,
    model='csv' selects legacy replay; every other old config selects ModelAgent.
    Adding agent types requires no new conditions here or in main.py.
    """
    kind = config.agent_type or ('csv' if config.model == 'csv' else 'model')
    if kind not in AGENTS:
        raise ValueError(f'Unknown agent_type {kind!r}. Registered: {sorted(AGENTS)}')
    agent = AGENTS[kind](config, base_directory=base_directory)
    if getattr(agent, 'agent_id', None) != config.agent_id:
        raise ValueError('Agent factory must preserve config.agent_id.')
    if not callable(getattr(agent, 'propose', None)):
        raise TypeError('Agent must implement propose(context).')
    return agent

def build_agents(configs, *, base_directory=''):
    """Construct an arbitrary mixture of registered stakeholder implementations."""
    ids = [cfg.agent_id for cfg in configs]
    if len(ids) != len(set(ids)):
        raise ValueError('Agent IDs must be unique.')
    return [build_agent(cfg, base_directory=base_directory) for cfg in configs]
