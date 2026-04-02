available quantizations on L4 (Ada Lovelace, sm_89):

L4 (Ada Lovelace, sm_89):

Format	Native Tensor Core	WMMA API	PTX mma.sync
FP8*	✅	❌ no wmma:: support	✅ direct PTX only
INT8	✅	✅	✅
INT4	✅	✅ wmma::experimental::precision::s4**	✅
FP4	    ❌	❌	❌ (Blackwell only)


*(e8m0/e4m3/e5m2)
** experimental, with know issues, we leave it