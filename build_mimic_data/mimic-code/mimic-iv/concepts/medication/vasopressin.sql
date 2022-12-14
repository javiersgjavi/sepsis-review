-- This query extracts dose+durations of vasopressin administration
select
  stay_id, linkorderid
  -- three rows in units/min, rest in units/hour
  -- the three rows in units/min look reasonable and fit with the patient course
  , CASE WHEN rateuom = 'units/min' THEN rate * 60.0
    ELSE rate END AS vaso_rate
  , amount as vaso_amount
  , starttime
  , endtime
from `physionet-data.mimic_icu.inputevents`
where itemid = 222315 -- vasopressin