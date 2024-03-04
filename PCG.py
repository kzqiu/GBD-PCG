import numpy as np

class PCG:
	def __init__(self, A, b, block_size, Nblocks, guess = None, options = {}):
		self.A = A
		self.b = b
		self.block_size = block_size
		self.Nblocks = Nblocks
		self.guess = guess
		if self.guess == None:
			self.guess = np.zeros(self.A.shape[0])
		self.options = options
		self.set_default_options(self.options)
		self.Pinv = self.compute_preconditioner(self.A, self.block_size, self.options['preconditioner_type'])

	def set_default_options(self, options):
		options.setdefault('exit_tolerance', 1e-6)
		options.setdefault('max_iter', 100)
		options.setdefault('DEBUG_MODE', False)
		options.setdefault('RETURN_TRACE', False)
		options.setdefault('preconditioner_type', 'BJ')
		options.setdefault('use_RK', False)
		self.validate_precon_type(options["preconditioner_type"])

	def update_A(self, A):
		self.A = A

	def update_b(self, b):
		self.b = b

	def update_guess(self, guess):
		self.guess = guess

	def update_exit_tolerance(self, tol):
		self.options["exit_tolerance"] = tol

	def update_max_iter(self, max_iter):
		self.options["max_iter"] = max_iter
	
	def update_preconditioner_type(self, type):
		self.validate_precon_type(type)
		self.options["preconditioner_type"] = type

	def update_DEBUG_MODE(self, mode):
		self.options["DEBUG_MODE"] = mode

	def update_RETURN_TRACE(self, mode):
		self.options["RETURN_TRACE"] = mode

	def validate_precon_type(self, precon_type):
		if not (precon_type in ['0', 'J', 'BJ', 'SS', 'SN']):
			print("Invalid preconditioner options are [0: none, J : Jacobi, BJ: Block-Jacobi, SS: Symmetric Stair]")
			exit()

	def invert_matrix(self, A):
		try:
			return np.linalg.inv(A)
		except:
			if self.options.get('DEBUG_MODE'):
				print("Warning singular matrix -- using Psuedo Inverse.")
			return np.linalg.pinv(A)

	def prk(self, A, b, Pinv, guess, options = {}):
		self.set_default_options(options)
		trace = []

		# initialize
		x = np.reshape(guess, (guess.shape[0],1))
		state_size = 2 # hardcoded for pend right now
		size = int(len(x)/state_size) 
		inds = list(range(size)) 
		# build normalized geometric-like probabilities
		prb = np.asarray([0.5**k for k in range(0, size)])
		prb /= np.sum(prb)
		# draw from inds accordingly
		rows = np.random.choice(inds, size=options['max_iter'], replace = True, p=prb)
		# apply precon?
		A = np.matmul(A,Pinv)
		err = b - np.matmul(A,x)
		# print("error norm", np.linalg.norm(err))
		# loop
		for iteration in range(options['max_iter']):
			# https://arxiv.org/pdf/1903.01806.pdf
			curr_row_start = state_size*rows[iteration]
			for row in range(curr_row_start,curr_row_start+state_size):
				numers = b[row] - np.matmul(A[row,:],x)
				denom = np.linalg.norm(A[row,:])
				right = numers/denom
				update = A[row,:]*right
				x += np.reshape(update, x.shape)
				err = b - np.matmul(A,x)
				# print("error norm", np.linalg.norm(err))
		# apply precon?
		x = np.matmul(x.T,Pinv).T
		return x

	def weighted_randomized_kaczmarz(self, A, b, Pinv, x0=None, p=10, max_iter=1000):
		# apply preconditioner
		A = np.dot(A, Pinv)
		m, n = A.shape
		epsilon = 1e-5
    
	    # Ensure that A has full rank
		# if np.linalg.matrix_rank(A) < n:
		# 	print(A.shape)
		# 	raise ValueError("Matrix A does not have full rank.")
    
	    # Initial solution
		if x0 is None:
			x0 = np.zeros((n, 1))
		xk = x0

	    # Pre-computation
		r0 = A @ xk
		Q = A @ A.T

	    # Iteration
		for _ in range(max_iter):
			Ax_minus_b_norm = np.linalg.norm(np.squeeze(A @ xk - b), ord=p)**p
        
	        # Compute probabilities for row selection
			prob = np.abs(A @ xk - b)**p / (Ax_minus_b_norm + epsilon)
			prob /= np.sum(prob)
			prob = np.squeeze(np.nan_to_num(prob))

			i = np.random.choice(m, p=prob)
        
	        # Compute lambda for the selected row
			lambda_ = ((b[i] - np.dot(A[i], xk)) / Q[i,i])[0]
        
	        # Update xk and rk
			xk = xk + lambda_ * A[i].reshape((m, 1))
			r0 = r0 + lambda_ * Q[:, i].reshape((m, 1))
        
	        # Convergence check (optional, depending on specific use-case)
			if np.linalg.norm(A @ xk - b) < 1e-3:
				break
    
		xk = np.dot(xk.T, Pinv).T
	
		return xk

	
	def mgrk(
		self,
		A: np.ndarray,
		b: np.ndarray,
		pinv: np.ndarray,
		alpha: float,
		beta: float,
		theta: float,
		x0=None,
		max_iter=10000,
		tol=1e-6,
	) -> np.ndarray:
		"""
		Solves the Ax = b system using the mGRK method from https://arxiv.org/pdf/2307.01988.pdf

		Parameters:
		- A: numpy array, the coefficient matrix A in Ax = b.
		- b: numpy array, the right-hand side vector in Ax = b.
		- alpha: float, the step size parameter.
		- beta: float, the momentum parameter.
		- theta: float, the parameter to adjust the greedy probability criterion.
		- x0: numpy array, initial guess for the solution.
		- max_iter: int, maximum number of iterations.
		- tol: float, tolerance for the stopping criterion.

		Returns:
		- x: numpy array, the approximate solution to Ax = b.
		"""
		A = np.dot(A, pinv)

		if x0 is None:
			x0 = np.zeros(A.shape[1])

		x = x0.copy()
		x_prev = x0.copy()
		b = b.T.squeeze()

		for _ in range(max_iter):
	        # Compute the residuals and determine the set Sk
			residuals = np.abs(np.dot(A, x) - b)

			if np.sum(residuals ** 2) < tol:
				break
			
	        # Simplified computation for gamma_k
			gamma_k = np.linalg.norm(A, ord="fro") ** 2
			criterion = (
				theta * np.max(residuals) ** 2
				+ (1 - theta) * np.linalg.norm(residuals) ** 2 / gamma_k
			)
			Sk = np.where(residuals**2 >= criterion)[0]

			if len(Sk) == 0:
				break  # All residuals are below the threshold

	        # Select ik from Sk based on some probability criterion (uniformly for simplicity)
			ik = np.random.choice(Sk)

	        # Update x using the mGRK formula
			a_ik = A[ik, :]
			numerator = np.dot(a_ik, x) - b[ik]
			denominator = np.linalg.norm(a_ik) ** 2

			x_next = x - (alpha * (numerator / denominator) * a_ik) + (beta * (x - x_prev))

			# if np.linalg.norm(x_next - x) < tol:
				# break  # Convergence criterion met

			x_prev = x
			x = x_next

		print(_)

		x = x.reshape(len(x), 1)
		x = np.dot(x.T, pinv).T

		return x

	def pcg(self, A, b, Pinv, guess, options = {}):
		self.set_default_options(options)
		trace = []

		if options['use_RK']:
			# return self.prk(A, b, Pinv, guess, options)
			# return self.weighted_randomized_kaczmarz(A, b, Pinv)
			return self.mgrk(A, b, Pinv, 0.8, 0.6, 0.5)
		# if options['only_precon']:
			# return np.matmul(Pinv,b)

		# initialize
		x = np.reshape(guess, (guess.shape[0],1))
		r = b - np.matmul(A, x)
		
		r_tilde = np.matmul(Pinv, r)
		p = r_tilde
		nu = np.matmul(r.transpose(), r_tilde)
		if options['DEBUG_MODE']:
			print("Initial nu[", nu, "]")
		if options['RETURN_TRACE']:
			trace = nu[0].tolist()
			trace2 = [np.linalg.norm(b - np.matmul(A, x))]
		# loop
		for iteration in range(options['max_iter']):
			Ap = np.matmul(A, p)
			alpha = nu / np.matmul(p.transpose(), Ap)
			r -= alpha * Ap
			x += alpha * p
			
			r_tilde = np.matmul(Pinv, r)
			nu_prime = np.matmul(r.transpose(), r_tilde)
			if options['RETURN_TRACE']:
				trace.append(nu_prime.tolist()[0][0])
				trace2.append(np.linalg.norm(b - np.matmul(A, x)))

			if abs(nu_prime) < options['exit_tolerance']:
				if options['DEBUG_MODE']:
					print(Pinv)
					print("Exiting with err[", abs(nu_prime), "]")
				break
			else:
				if options['DEBUG_MODE']:
					print("Iter[", iteration, "] with err[", abs(nu_prime), "]")
			
			beta = nu_prime / nu
			p = r_tilde + beta * p
			nu = nu_prime

		# print(iteration)
		if options['RETURN_TRACE']:
			trace = list(map(abs,trace))
			return x, (trace, trace2)
		else:
			return x

	def compute_preconditioner(self, A, block_size, preconditioner_type):
		if preconditioner_type == "0": # null aka identity
			return np.identity(A.shape[0])

		if preconditioner_type == "J": # Jacobi aka Diagonal
			return self.invert_matrix(np.diag(np.diag(A)))

		elif preconditioner_type == "BJ": # Block-Jacobi
			n_blocks = int(A.shape[0] / block_size)
			Pinv = np.zeros(A.shape)
			for k in range(n_blocks):
				rc_k = k*block_size
				rc_kp1 = rc_k + block_size
				Pinv[rc_k:rc_kp1, rc_k:rc_kp1] = self.invert_matrix(A[rc_k:rc_kp1, rc_k:rc_kp1])

			return Pinv

		elif preconditioner_type == "SS": # Symmetric Stair (for blocktridiagonal of blocksize nq+nv)
			n_blocks = int(A.shape[0] / block_size)
			Pinv = np.zeros(A.shape)
			# compute stair inverse
			for k in range(n_blocks):
				# compute the diagonal term
				Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size] = \
					self.invert_matrix(A[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size])
				if np.mod(k, 2): # odd block includes off diag terms
					# Pinv_left_of_diag_k = -Pinv_diag_k * A_left_of_diag_k * -Pinv_diag_km1
					Pinv[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size] = \
						-np.matmul(Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size], \
								   np.matmul(A[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size], \
									  		 Pinv[(k-1)*block_size:k*block_size, (k-1)*block_size:k*block_size]))
				elif k > 0: # compute the off diag term for previous odd block (if it exists)
					# Pinv_right_of_diag_km1 = -Pinv_diag_km1 * A_right_of_diag_km1 * -Pinv_diag_k
					Pinv[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size] = \
						-np.matmul(Pinv[(k-1)*block_size:k*block_size, (k-1)*block_size:k*block_size], \
								   np.matmul(A[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size], \
											 Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size]))
			# make symmetric
			for k in range(n_blocks):
				if np.mod(k, 2): # copy from odd blocks
					# always copy up the left to previous right
					Pinv[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size] = \
						Pinv[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size].transpose()
					# if not last block copy right to next left
					if k < n_blocks - 1:
						Pinv[(k+1)*block_size:(k+2)*block_size, k*block_size:(k+1)*block_size] = \
							Pinv[k*block_size:(k+1)*block_size, (k+1)*block_size:(k+2)*block_size].transpose()
			return Pinv

		elif preconditioner_type == "SN": #[0] == "S" and preconditioner_type[1:].isnumeric(): # Stair to the Nth (for blocktridiagonal of blocksize nq+nv)
			series_levels = 100#int(preconditioner_type[1:])
			n_blocks = int(A.shape[0] / block_size)
			Pinv = np.zeros(A.shape)
			Q = np.zeros(A.shape)
			# compute stair inverse
			for k in range(n_blocks):
				# compute the diagonal term
				Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size] = \
					self.invert_matrix(A[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size])
				if np.mod(k, 2): # odd block includes off diag terms
					# Pinv_left_of_diag_k = -Pinv_diag_k * A_left_of_diag_k * -Pinv_diag_km1
					Pinv[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size] = \
						-np.matmul(Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size], \
								   np.matmul(A[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size], \
									  		 Pinv[(k-1)*block_size:k*block_size, (k-1)*block_size:k*block_size]))
				elif k > 0: # compute the off diag term for previous odd block (if it exists)
					# Pinv_right_of_diag_km1 = -Pinv_diag_km1 * A_right_of_diag_km1 * -Pinv_diag_k
					Pinv[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size] = \
						-np.matmul(Pinv[(k-1)*block_size:k*block_size, (k-1)*block_size:k*block_size], \
								   np.matmul(A[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size], \
											 Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size]))

			# form the remainder
			for k in range(n_blocks):
				# start to the right then down -- so only exists for even block-row and block-column minus and plus 1
				if (k % 2) == 0:
					start = k*block_size
					if k > 0: # block-column minus 1 exists
						Q[start:start+block_size,start-block_size:start] = \
						-A[start:start+block_size,start-block_size:start]
					if k < n_blocks - 1: # block-column plus 1 exists
						Q[start:start+block_size,start+block_size:start+2*block_size] = \
						-A[start:start+block_size,start+block_size:start+2*block_size]
			# compute the series Final_Pinv = (SUM H^k) * Pinv

			H = np.matmul(Pinv,Q)
			sumterm = np.eye(A.shape[0])
			base = np.eye(A.shape[0])

			# Ainv = np.linalg.inv(A)
			# err = Ainv - np.matmul(sumterm,Pinv)
			# print("At level 0 err is ", np.linalg.norm(err))
			
			for series_level in range(series_levels):
				base = np.matmul(base,H)
				sumterm += base
				# err = Ainv - np.matmul(sumterm,Pinv)
				# print("At level X err is ", np.linalg.norm(err))
			res = np.matmul(sumterm,Pinv)

			# err = Ainv - res
			# print("At end err is ", np.linalg.norm(err))
			return (res.T + res)/2

	def solve(self):
	    return self.pcg(self.A, self.b, self.Pinv, self.guess, self.options)
