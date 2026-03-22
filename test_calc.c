int CLI_calc_expr_double(const char *expr, double *res);
int main() { double res; CLI_calc_expr_double("where(min(2,4)>1, 100, -100)", &res); printf("%f
", res); return 0; }
