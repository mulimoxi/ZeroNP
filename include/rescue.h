#pragma once
ZERONPWork* init_work_RESCUE
(
    ZERONPWork* w_old,
    ZERONPSettings* stgs
);
ZERONPWork* ZERONP_RESCUE_init(
    ZERONPWork* w_old,
    ZERONPSettings* stgs
    );
zeronp_int update_work_rescue
(
    ZERONPWork* w,
    ZERONPSettings* stgs
);
zeronp_int ZERONP_RESCUE_solve
(
    ZERONPWork* w,
    ZERONPSettings* stgs,
    ZERONPInfo* info
    );
ZERONPWork* ZERONP_RESCUE
(
    ZERONPWork* w_old,
    ZERONPSettings* stgs,
    ZERONPInfo* info // record running time
    );