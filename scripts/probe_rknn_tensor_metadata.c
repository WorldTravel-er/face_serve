/*
 * Query deployable RKNN tensor metadata through the documented C Runtime API.
 * Build on RK3588 with:
 *   gcc -O2 -Wall -Wextra -o probe_rknn_tensor_metadata \
 *       scripts/probe_rknn_tensor_metadata.c -lrknnrt
 * Run on the same librknnrt.so used by the service:
 *   ./probe_rknn_tensor_metadata /opt/face-serve/models/rknn/yolo-fp.rknn \
 *       > yolo-fp.probe.json
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rknn_api.h"

/* Minimal self-contained SHA-256 so probe identity has no extra library. */
typedef struct { uint32_t h[8]; uint64_t bits; unsigned char block[64]; size_t used; } sha256_ctx;
static const uint32_t sha_k[64] = {0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define ROR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha_block(sha256_ctx *c, const unsigned char *p) {
  uint32_t w[64], a,b,d,e,f,g,h,t1,t2; uint32_t cc; int i;
  for (i=0;i<16;i++) w[i]=((uint32_t)p[4*i]<<24)|((uint32_t)p[4*i+1]<<16)|((uint32_t)p[4*i+2]<<8)|p[4*i+3];
  for (;i<64;i++) w[i]=(ROR(w[i-15],7)^ROR(w[i-15],18)^(w[i-15]>>3))+w[i-16]+(ROR(w[i-2],17)^ROR(w[i-2],19)^(w[i-2]>>10))+w[i-7];
  a=c->h[0]; b=c->h[1]; cc=c->h[2]; d=c->h[3]; e=c->h[4]; f=c->h[5]; g=c->h[6]; h=c->h[7];
  for (i=0;i<64;i++) { t1=h+(ROR(e,6)^ROR(e,11)^ROR(e,25))+((e&f)^((~e)&g))+sha_k[i]+w[i]; t2=(ROR(a,2)^ROR(a,13)^ROR(a,22))+((a&b)^(a&cc)^(b&cc)); h=g; g=f; f=e; e=d+t1; d=cc; cc=b; b=a; a=t1+t2; }
  c->h[0]+=a; c->h[1]+=b; c->h[2]+=cc; c->h[3]+=d; c->h[4]+=e; c->h[5]+=f; c->h[6]+=g; c->h[7]+=h;
}
static void sha_update(sha256_ctx *c, const unsigned char *p, size_t n) { while(n--) { c->block[c->used++]=*p++; c->bits+=8; if(c->used==64) { sha_block(c,c->block); c->used=0; } } }
static void sha_final(sha256_ctx *c, char out[65]) { unsigned char digest[32]; int i; c->block[c->used++]=0x80; if(c->used>56) { while(c->used<64)c->block[c->used++]=0; sha_block(c,c->block); c->used=0; } while(c->used<56)c->block[c->used++]=0; for(i=7;i>=0;i--)c->block[c->used++]=(unsigned char)(c->bits>>(8*i)); sha_block(c,c->block); for(i=0;i<8;i++){digest[4*i]=(unsigned char)(c->h[i]>>24);digest[4*i+1]=(unsigned char)(c->h[i]>>16);digest[4*i+2]=(unsigned char)(c->h[i]>>8);digest[4*i+3]=(unsigned char)c->h[i];} for(i=0;i<32;i++)sprintf(out+2*i,"%02x",digest[i]); out[64]='\0'; }
static int sha256_file_hex(const char *path, char out[65]) { FILE *f=fopen(path,"rb"); unsigned char buf[8192]; size_t n; sha256_ctx c={{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19},0,{0},0}; if(!f)return 1; while((n=fread(buf,1,sizeof(buf),f))>0)sha_update(&c,buf,n); if(ferror(f)){fclose(f);return 1;} fclose(f); sha_final(&c,out); return 0; }
static int read_model_file(const char *path, unsigned char **buffer, size_t *length) { FILE *f=fopen(path,"rb"); long size; size_t got; if(!f)return 1; if(fseek(f,0,SEEK_END)||((size=ftell(f))<1)||fseek(f,0,SEEK_SET)){fclose(f);return 1;} *buffer=malloc((size_t)size); if(!*buffer){fclose(f);return 1;} got=fread(*buffer,1,(size_t)size,f); fclose(f); if(got!=(size_t)size){free(*buffer);*buffer=NULL;return 1;} *length=got; return 0; }

static const char *layout_name(rknn_tensor_format fmt) {
  switch (fmt) {
    case RKNN_TENSOR_NCHW: return "nchw";
    case RKNN_TENSOR_NHWC: return "nhwc";
    case RKNN_TENSOR_NC1HWC2: return "nc1hwc2";
    default: return "undefined";
  }
}

static const char *dtype_name(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32: return "float32";
    case RKNN_TENSOR_FLOAT16: return "float16";
    case RKNN_TENSOR_INT8: return "int8";
    case RKNN_TENSOR_UINT8: return "uint8";
    case RKNN_TENSOR_INT16: return "int16";
    case RKNN_TENSOR_UINT16: return "uint16";
    case RKNN_TENSOR_INT32: return "int32";
    case RKNN_TENSOR_UINT32: return "uint32";
    default: return "unsupported";
  }
}

static void print_json_string(const char *value) {
  const unsigned char *p = (const unsigned char *)value;
  putchar('"');
  for (; *p; ++p) {
    switch (*p) {
      case '"': fputs("\\\"", stdout); break;
      case '\\': fputs("\\\\", stdout); break;
      case '\n': fputs("\\n", stdout); break;
      case '\r': fputs("\\r", stdout); break;
      case '\t': fputs("\\t", stdout); break;
      default:
        if (*p < 0x20) printf("\\u%04x", *p);
        else putchar(*p);
    }
  }
  putchar('"');
}

static int print_tensors(rknn_context ctx, unsigned int count, int is_input) {
  unsigned int i, j;
  for (i = 0; i < count; ++i) {
    rknn_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    int query = rknn_query(ctx, is_input ? RKNN_QUERY_INPUT_ATTR : RKNN_QUERY_OUTPUT_ATTR,
                           &attr, sizeof(attr));
    if (query != RKNN_SUCC) {
      fprintf(stderr, "rknn_query(%s,%u) failed: %d\n", is_input ? "input" : "output", i, query);
      return 1;
    }
    if (strcmp(dtype_name(attr.type), "unsupported") == 0) {
      fprintf(stderr, "unsupported tensor dtype for %s[%u]\n", is_input ? "input" : "output", i);
      return 1;
    }
    printf("%s{\"name\":", i ? "," : "");
    print_json_string(attr.name);
    printf(",\"layout\":");
    print_json_string(layout_name(attr.fmt));
    printf(",\"shape\":[");
    for (j = 0; j < attr.n_dims; ++j) printf("%s%u", j ? "," : "", attr.dims[j]);
    printf("],\"dtype\":");
    print_json_string(dtype_name(attr.type));
    printf("}");
  }
  return 0;
}

int main(int argc, char **argv) {
  rknn_context ctx = 0;
  rknn_input_output_num io_num;
  int ret;
  char model_sha256[65];
  unsigned char *model_data = NULL;
  size_t model_length = 0;
  if (argc != 2) {
    fprintf(stderr, "usage: %s model.rknn\n", argv[0]);
    return 2;
  }
  if (sha256_file_hex(argv[1], model_sha256)) { fprintf(stderr, "cannot SHA-256 model: %s\n", argv[1]); return 1; }
  if (read_model_file(argv[1], &model_data, &model_length)) { fprintf(stderr, "cannot read model: %s\n", argv[1]); return 1; }
  ret = rknn_init(&ctx, model_data, model_length, 0, NULL);
  if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_init failed: %d\n", ret); free(model_data); return 1; }
  memset(&io_num, 0, sizeof(io_num));
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_query(io_num) failed: %d\n", ret); rknn_destroy(ctx); free(model_data); return 1; }
  printf("{\"rknn_path\":");
  print_json_string(argv[1]);
  printf(",\"rknn_sha256\":");
  print_json_string(model_sha256);
  printf(",\"inputs\":[");
  if (print_tensors(ctx, io_num.n_input, 1)) { rknn_destroy(ctx); free(model_data); return 1; }
  printf("],\"outputs\":[");
  if (print_tensors(ctx, io_num.n_output, 0)) { rknn_destroy(ctx); free(model_data); return 1; }
  printf("]}\n");
  ret = rknn_destroy(ctx);
  free(model_data);
  if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_destroy failed: %d\n", ret); return 1; }
  return 0;
}
