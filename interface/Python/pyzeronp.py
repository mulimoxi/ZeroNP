from ctypes import c_double, c_int, CDLL, CFUNCTYPE, POINTER
import sys
import numpy
import ZERONP_CONST as CONST
from os import path
from glob import glob, escape
import platform
import os


class ZERONP:
    def __init__(
        self,
        prob: dict,
        op: dict = None,
        cost=None,
        grad_fun=None,
        hess_fun=None,
        l: numpy.ndarray = None,
        h: numpy.ndarray = None,
    ):
        self.prob = prob
        self.op = op
        self.l = l
        self.h = h
        self.cost = cost

        self.prob_c = dict()
        self.op_c = None
        self.l_c = None
        self.h_c = None
        self.get_input()

        root = path.abspath(path.dirname(__file__))

        if platform.system() == "Linux":
            lib_path = glob(escape(root) + "/*.so")[0]
            try:
                self.zeronp_lib = CDLL(lib_path)
            except:
                print("Error in loading libzeronp.so on Linux")

        elif platform.system() == "Windows":
            lib_path = glob(escape(root) + "/*.dll")[0]

            try:
                self.zeronp_lib = CDLL(lib_path)
            except:
                print("Error in loading zeronp.dll on Windows")
        else:
            lib_path = glob(escape(root) + "/*.dylib")[0]
            try:
                self.zeronp_lib = CDLL(lib_path)
            except:
                print("Error in loading libzeronp.dylib on MacOS")

        self.python_c_zeronp = self.zeronp_lib.ZERONP_C
        self.python_c_zeronp.restype = None
        self.cost = cost
        self.grad_fun = grad_fun
        self.hess_fun = hess_fun
        np = self.prob["np"]
        nic = self.prob["nic"]
        nec = self.prob["nec"]
        nc = nic + nec
        len_h = (np + nic) ** 2 if self.op["bfgs"] == 1 else 1

        self.python_c_zeronp.argtypes = [
            c_double * self.prob["nic"],
            c_double * self.prob["nic"],
            c_double * self.prob["np"],
            c_double * self.prob["np"],
            c_int * 2,
            c_int * 2,
            c_double * self.prob["nic"],
            c_double * self.prob["np"],
            c_double * len(self.op),
            c_double * self.prob["nc"],
            c_double * (self.prob["np"] + self.prob["nic"]) ** 2,
            c_int,
            c_int,
            c_int,
            # c_double * 9,
            c_double * 10,
            c_double * np,
            c_double * np,
            c_double * max(nic, 1),
            c_double * (self.op["max_iter"] + 1),
            c_double * (self.op["max_iter"] + 1),
            c_double * max(nc, 1),
            c_double * len_h,
        ]

        # self.COST_TEMPLE = CFUNCTYPE(None, POINTER(c_double), POINTER(c_double), c_int)
        self.COST_TEMPLE = CFUNCTYPE(None, POINTER(c_double), POINTER(c_double))

        # self.COST_IN_PYTHON = self.COST_TEMPLE(cost)
        self.COST_IN_PYTHON = self.COST_TEMPLE(self.my_cost)

        self.GRAD_IN_PYTHON = None
        self.HESS_IN_PYTHON = None

        if grad_fun is not None:
            GRAD_TEMPLE = CFUNCTYPE(None, POINTER(c_double), POINTER(c_double))
            self.GRAD_IN_PYTHON = GRAD_TEMPLE(self.my_grad)

        if hess_fun is not None:
            HESS_TEMPLE = CFUNCTYPE(None, POINTER(c_double), POINTER(c_double))
            self.HESS_IN_PYTHON = HESS_TEMPLE(self.my_hess)

        self.zeronp_lib.def_python_callback(
            self.COST_IN_PYTHON, self.GRAD_IN_PYTHON, self.HESS_IN_PYTHON
        )

    # def my_cost(self, x, y, nfeval):
    def my_cost(self, x, y):
        np_x = numpy.ctypeslib.as_array(x, shape=(self.prob["np"],))
        # result = self.cost(np_x, nfeval)
        result = self.cost(np_x)
        for i in range(len(result)):
            y[i] = result[i]
        return

    def my_grad(self, x, y):
        np_x = numpy.ctypeslib.as_array(x, shape=(self.prob["np"],))
        result = self.grad_fun(np_x)
        for i in range(len(result)):
            y[i] = result[i]
        return

    def my_hess(self, x, y):
        np_x = numpy.ctypeslib.as_array(x, shape=(self.prob["np"],))
        result = self.hess_fun(np_x)
        for i in range(len(result)):
            y[i] = result[i]
        return

    def get_input(self):
        self.fulfill_prob()
        self.get_settings(self.prob["np"])
        self.check_prob()
        self.check_l_h()

        # self.prob["Ipc"] = [1, 1]
        # self.prob["Ipb"] = [1, 1]

        self.prob_py2c()

        return

    def check_l_h(self):
        """
        check l and h
        """

        # l
        if self.l is None:
            self.l = numpy.zeros(self.prob["nc"])

        # h
        if self.h is None:
            # self.h = numpy.eye(self.prob['nc']+self.prob['nic'])
            self.h = numpy.eye(self.prob["np"] + self.prob["nic"])

        return

    def fulfill_prob(self):
        """
        fulfill prob dict if key is missing

        after this func, prob should contain
          - Ipc
          - Ipb
          - ibl
          - ibu
          - pbl
          - pbu
          - ib0
          - p0
          - np
          - nic
          - nec
          - nc
        """

        # init np, nic, nec, nc
        np = 0  # dim of variables
        nic = 0  # dim of inequality constraints
        nec = 0  # dim of equality constraints
        nc = 0  # dim of constraints

        # Ipc
        self.prob["Ipc"] = [0, 0]

        # Ipb
        self.prob["Ipb"] = [0, 0]

        # ibl
        if "ibl" not in self.prob:
            self.prob["ibl"] = None
        else:
            nic = len(self.prob["ibl"])

        # ibu
        if "ibu" not in self.prob:
            self.prob["ibu"] = None
        else:
            nic = len(self.prob["ibu"])

        # pbl
        if "pbl" not in self.prob:
            self.prob["pbl"] = None
        else:
            np = len(self.prob["pbl"])
            self.prob["Ipc"][0] = 1
            self.prob["Ipb"][0] = 1

        # pbu
        if "pbu" not in self.prob:
            self.prob["pbu"] = None
        else:
            np = len(self.prob["pbu"])
            self.prob["Ipc"][1] = 1
            self.prob["Ipb"][0] = 1

        # ib0
        if "ib0" not in self.prob:
            self.prob["ib0"] = None
        else:
            nic = len(self.prob["ib0"])

        # p0
        if "p0" not in self.prob:
            self.prob["p0"] = None
        else:
            np = len(self.prob["p0"])

        # This is the missing part that caused memory bug
        # first call cost to get the dimension
        # first_result = self.cost(numpy.array([0] * np), 1)
        first_result = self.cost(numpy.array([0] * np))
        m = len(first_result)
        nec = m - 1 - nic
        #########################################################

        # record np, nic, nec, nc
        nc = nic + nec
        self.prob["np"] = np
        self.prob["nic"] = nic
        self.prob["nec"] = nec
        self.prob["nc"] = nc

        if self.prob["Ipb"][0] + nic >= 0.5:
            self.prob["Ipb"][1] = 1

        return

    def check_prob(self):
        """
        check if prob is legal
        and init those None items
        """

        prob_status = True  # True if prob is legal

        # ibl, ibu and ib0
        if self.prob["nic"] > 0:
            # init ibl, ibu
            if self.prob["ibl"] is None:
                self.prob["ibl"] = numpy.full(self.prob["nic"], -CONST.INFINITY)
            if self.prob["ibu"] is None:
                self.prob["ibu"] = numpy.full(self.prob["nic"], CONST.INFINITY)
            # check and set ib0
            if self.check_bound(lb=self.prob["ibl"], ub=self.prob["ibu"]) is False:
                prob_status = False
                raise ValueError("Inequality bound error!")
            if (
                self.check_var_bound(
                    x=self.prob["ib0"], lb=self.prob["ibl"], ub=self.prob["ibu"]
                )
                is False
            ):
                self.prob["ib0"] = self.cal_ib0(
                    ibl=self.prob["ibl"], ibu=self.prob["ibu"]
                )

        # pbl, pbu and p0
        if self.prob["np"] > 0:
            # init pbl, pbu
            if self.prob["pbl"] is None:
                self.prob["pbl"] = numpy.full(self.prob["np"], -CONST.INFINITY)
            if self.prob["pbu"] is None:
                self.prob["pbu"] = numpy.full(self.prob["np"], CONST.INFINITY)
            # check and set p0
            if self.check_bound(lb=self.prob["pbl"], ub=self.prob["pbu"]) is False:
                prob_status = False
                raise ValueError("Variable bound error!")
            if (
                self.check_var_bound(
                    x=self.prob["p0"], lb=self.prob["pbl"], ub=self.prob["pbu"]
                )
                is False
            ):
                self.prob["p0"] = self.cal_p0(
                    pbl=self.prob["pbl"], pbu=self.prob["pbu"]
                )
        else:
            prob_status = False
            raise ValueError("Dim of variables is 0!")

        return prob_status

    def check_var_bound(self, x: numpy.ndarray, lb: numpy.ndarray, ub: numpy.ndarray):
        """
        check
          - x is not None
          - len(x) == len(lb) == len(ub)
          - lb < x < ub
        """

        var_bound_status = True

        if x is None:
            var_bound_status = False
        else:
            if len(lb) == len(ub) and len(lb) == len(x):
                if self.check_strict_in_bound(x=x, lb=lb, ub=ub) is False:
                    var_bound_status = False
            else:
                var_bound_status = False
        return var_bound_status

    def check_bound(self, lb: numpy.ndarray, ub: numpy.ndarray):
        """
        check lb < ub
        """

        bound_status = True

        if numpy.any(lb > ub):
            bound_status = False

        return bound_status

    def check_strict_in_bound(
        self, x: numpy.ndarray, lb: numpy.ndarray, ub: numpy.ndarray
    ):
        """
        check lb < x < ub
        """

        inbound_status = True

        if numpy.any(x <= lb) or numpy.any(x >= ub):
            inbound_status = False

        return inbound_status

    def cal_ib0(self, ibl: numpy.ndarray, ibu: numpy.ndarray):
        ib0 = (ibl + ibu) / 2
        return ib0

    def cal_p0(self, pbl: numpy.ndarray, pbu: numpy.ndarray):
        p0 = (pbl + pbu) / 2
        return p0

    def prob_py2c(self):
        """
        convert self.prob to self.prob_c

        zeronp_float* ibl,
        zeronp_float* ibu,
        zeronp_float* pbl,
        zeronp_float* pbu,
        zeronp_float* Ipc,
        zeronp_float* Ipb,
        zeronp_float* ib0,
        zeronp_float* p,
        zeronp_float* op,
        zeronp_float* l,
        zeronp_float* h,
        zeronp_int np,
        zeronp_int nic,
        zeronp_int nec,
        """

        # np, nic, nec, nc
        self.prob_c["np"] = c_int(self.prob["np"])
        self.prob_c["nic"] = c_int(self.prob["nic"])
        self.prob_c["nec"] = c_int(self.prob["nec"])
        self.prob_c["nc"] = c_int(self.prob["nc"])

        # ibl, ibu, ib0
        self.prob_c["ibl"] = (
            (c_double * self.prob["nic"])(*self.prob["ibl"])
            if self.prob["nic"] > 0
            else (c_double * 0)()
        )
        self.prob_c["ibu"] = (
            (c_double * self.prob["nic"])(*self.prob["ibu"])
            if self.prob["nic"] > 0
            else (c_double * 0)()
        )
        self.prob_c["ib0"] = (
            (c_double * self.prob["nic"])(*self.prob["ib0"])
            if self.prob["nic"] > 0
            else (c_double * 0)()
        )

        # pbl, pbu, p0
        self.prob_c["pbl"] = (c_double * self.prob["np"])(*self.prob["pbl"])
        self.prob_c["pbu"] = (c_double * self.prob["np"])(*self.prob["pbu"])
        self.prob_c["p"] = (c_double * self.prob["np"])(*self.prob["p0"])
        # Ipc, Ipb
        self.prob_c["Ipc"] = (c_int * 2)(*self.prob["Ipc"])
        self.prob_c["Ipb"] = (c_int * 2)(*self.prob["Ipb"])
        # l
        self.l_c = (
            (c_double * self.prob["nc"])(*self.l)
            if self.prob["nc"] > 0
            else (c_double * 0)()
        )

        # h
        self.h_c = (c_double * (self.prob["np"] + self.prob["nic"]) ** 2)(
            *self.h.reshape(-1)
        )

        # op
        # self.op_c = [c_double(self.op['rho']), c_double(self.op['max_iter']), c_double(
        #     self.op['min_iter']), c_double(self.op['delta']), c_double(self.op['tol'])]

        return

    def print_info_prefix(self, line=1):
        if line == 1:
            print("ZERONP--> ", end="")
        elif line == 0:
            print("         ", end="")
        elif line == -1:
            print()
        return

    def default_settings(self, np):
        settings = {
            "rho": 1.0,
            "pen_l1": 1,
            "max_iter": 50,
            "min_iter": 10,
            "max_iter_rescue": 50,
            "min_iter_rescue": 10,
            "delta": 1.0,
            "tol": 1e-4,
            "tol_con": 1e-3,
            "ls_time": 10,
            "batchsize": max(min(50, int(np / 4)), 1),
            "tol_restart": 1.0,
            "re_time": 5,
            "delta_end": 1e-5,
            "maxfev": 500 * np,
            "noise": 1,
            "qpsolver": 1,
            "scale": 1,
            "bfgs": 1,
            "rs": 0,
            "grad": 1,
            "k_i": 3.0,
            "k_r": 9,
            "c_r": 10.0,
            "c_i": 30.0,
            "ls_way": 1,
            # 'rescue': 1,
            "rescue": 0,
            "drsom": 0,
            "cen_diff": 0,
            "gd_step": 1e-1,
            "step_ratio": 1.0 / 3,
            "verbose": 1,
        }
        return settings

    def get_settings(self, np):
        settings = self.default_settings(np)

        if self.op is not None:
            for key in self.op.keys():
                settings[key] = self.op[key]

        # if self.op['noise'] == 0 and 'delta' not in self.op.keys():
        if settings["noise"] == 0 and (
            self.op is None or "delta" not in self.op.keys()
        ):
            settings["delta"] = 1e-5

        if self.prob["nec"] >= self.prob["np"]:
            settings["rescue"] = 1

        if self.op is None or "max_iter_rescue" not in self.op.keys():
            settings["max_iter_rescue"] = settings["max_iter"]

        if self.op is None or "min_iter_rescue" not in self.op.keys():
            settings["min_iter_rescue"] = settings["min_iter"]

        if self.op is not None and "rs" not in self.op.keys():
            settings["grad"] = 1

        if settings["rescue"] == 1:
            settings["min_iter"] = 1
        # self.op = list(settings.values())
        self.op = settings

    def run(self):
        """
        input:
        zeronp_float* ibl,
        zeronp_float* ibu,
        zeronp_float* pbl,
        zeronp_float* pbu,
        zeronp_float* Ipc,
        zeronp_float* Ipb,
        zeronp_float* ib0,
        zeronp_float* p,
        zeronp_float* op,
        zeronp_float* l,
        zeronp_float* h,
        zeronp_int np,
        zeronp_int nic,
        zeronp_int nec,

        output:
        zeronp_float* scalars,
        zeronp_float* p_out,
        zeronp_float* best_fea_p,
        zeronp_float* ic,
        zeronp_float* jh,
        zeronp_float* ch,
        zeronp_float* l_out,
        zeronp_float* h_out,
        """
        # self.zeronp_lib.def_python_callback(self.COST_IN_PYTHON)
        self.zeronp_lib.def_python_callback(
            self.COST_IN_PYTHON, self.GRAD_IN_PYTHON, self.HESS_IN_PYTHON
        )

        n_p = self.prob["np"]
        nic = self.prob["nic"]
        nec = self.prob["nec"]
        nc = nic + nec

        # create c arrays for output
        scalars = (c_double * 10)()

        p_out = (c_double * n_p)()
        best_fea_p = (c_double * n_p)()
        ic = (c_double * max(nic, 1))()
        jh = (c_double * (self.op["max_iter"] + 1))()
        ch = (c_double * (self.op["max_iter"] + 1))()
        count_h = (c_double * (self.op["max_iter"] + 1))()
        l_out = (c_double * max(nc, 1))()
        len_h = (n_p + nic) ** 2 if self.op["bfgs"] == 1 else 1
        h_out = (c_double * len_h)()
        self.op = list(self.op.values())
        self.op_c = (c_double * len(self.op))(*self.op)

        self.python_c_zeronp(
            self.prob_c["ibl"],
            self.prob_c["ibu"],
            self.prob_c["pbl"],
            self.prob_c["pbu"],
            self.prob_c["Ipc"],
            self.prob_c["Ipb"],
            self.prob_c["ib0"],
            self.prob_c["p"],
            self.op_c,
            self.l_c,
            self.h_c,
            self.prob_c["np"],
            self.prob_c["nic"],
            self.prob_c["nec"],
            scalars,
            p_out,
            best_fea_p,
            ic,
            jh,
            ch,
            l_out,
            h_out,
            count_h,
        )

        solution = {
            "iter": int(scalars[0]),
            "count_cost": int(scalars[1]),
            "count_grad": int(scalars[2]),
            "count_hess": int(scalars[3]),
            "constraint": scalars[4],
            "restart_time": int(scalars[5]),
            "obj": scalars[6],
            "status": int(scalars[7]),
            "solve_time": scalars[8],
            "qpsolver_time": scalars[9],
            "p": numpy.array(p_out),
            "best_fea_p": numpy.array(best_fea_p),
            "ic": numpy.array(ic),
            "jh": numpy.array(jh[: int(scalars[0])]),
            "ch": numpy.array(ch[: int(scalars[0])]),
            "count_h": numpy.array(count_h[: int(scalars[0])]),
            "l": numpy.array(l_out),
            "h": numpy.array(l_out),
        }

        return solution
