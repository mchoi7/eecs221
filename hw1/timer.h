#if defined (__cplusplus) // used to declare/initialize/include things when they are also defined in other files as well
extern "C" {
#endif

struct stopwatch_t * stopwatch_create (void);
void stopwatch_destroy (struct stopwatch_t* T);
void stopwatch_init (void);

void stopwatch_start (struct stopwatch_t* T);

long double stopwatch_stop (struct stopwatch_t* T);


#if defined (__cplusplus)
} // extern "C"
#endif
