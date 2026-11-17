#ifndef BN_NATIVE_SCHEDULE_METRICS_H
#define BN_NATIVE_SCHEDULE_METRICS_H
#include <math.h>
#include <stddef.h>
#define BN_NATIVE_SERVING_REQUESTS 100
/* Nearest-rank quantiles of completed-request samples; preserve input and
 * reject missing/invalid observations. Small n remains small n, including p99. */
typedef struct {size_t count;double minimum,p50,p95,p99,maximum;} BnNativeServingDistribution;
static inline int native_serving_distribution(const double *values,size_t count,BnNativeServingDistribution *out) {
    if(!values||!out||!count||count>BN_NATIVE_SERVING_REQUESTS)return -1;
    double sorted[BN_NATIVE_SERVING_REQUESTS];
    for(size_t i=0;i<count;i++) {
        if(!isfinite(values[i])||values[i]<0)return -1;
        size_t j=i;while(j&&sorted[j-1]>values[i]){sorted[j]=sorted[j-1];j--;}
        sorted[j]=values[i];
    }
    *out=(BnNativeServingDistribution){count,sorted[0],sorted[(count*50+99)/100-1],
        sorted[(count*95+99)/100-1],sorted[(count*99+99)/100-1],sorted[count-1]};return 0;
}
#endif
