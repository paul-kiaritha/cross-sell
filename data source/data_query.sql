SELECT DISTINCT Cust,
                P_Group,
                P_Age,
                C_Age,
                ROUND((P_Age / C_Age), 2) AS P_Rating
FROM (
    SELECT A.Cust,
           S_Unit,
           CASE
               WHEN Cat IN (1202, 1203, 21001) THEN 'FIXED.DEPOSIT'
               WHEN Cat IN (6050, 1204) THEN 'CALL.DEPOSIT'
               WHEN C.P_Group = 'PERSONAL.LN' THEN 'MOBILE.DIGITAL.LENDING'
               WHEN Cat BETWEEN 1000 AND 1999 THEN 'CURRENT.ACCOUNTS'
               WHEN Cat BETWEEN 2000 AND 2999 THEN 'VOSTROS'
               WHEN Cat = '3138' THEN 'IPF'
               WHEN Cat BETWEEN 3000 AND 3999 THEN C.P_Group
               WHEN Cat BETWEEN 4000 AND 4999 THEN 'CURRENT.ACCOUNTS'
               WHEN Cat BETWEEN 5000 AND 5999 THEN 'NOSTRO'
               WHEN Cat BETWEEN 6000 AND 6999 THEN 'SAVINGS.ACCOUNTS'
               WHEN Cat BETWEEN 7000 AND 7999 THEN 'PROVISIONS'
               WHEN Cat BETWEEN 8000 AND 8999 THEN 'CONTINGENT'
               WHEN Cat BETWEEN 9000 AND 9999 THEN 'CONTINGENT'
               ELSE Cat
           END AS P_Group,
           CASE
               WHEN ROUND((TRUNC(SYSDATE) - TO_DATE(A.Op_Date, 'yyyymmdd')) / 365) = 0 THEN 1
               ELSE ROUND((TRUNC(SYSDATE) - TO_DATE(A.Op_Date, 'yyyymmdd')) / 365)
           END P_Age,
           ROUND((TRUNC(TO_DATE(SYSDATE)) - NVL(NVL(TO_DATE(CASE
                                                            WHEN B_Date LIKE '11%' THEN '19' || SUBSTR(B_Date, 3, 6)
                                                            WHEN B_Date LIKE '13%' THEN '19' || SUBSTR(B_Date, 3, 6)
                                                            WHEN B_Date LIKE '10%' THEN '19' || SUBSTR(B_Date, 3, 6)
                                                            ELSE B_Date
                                                        END, 'YYYYMMDD'),
                                             TO_DATE(CASE
                                                          WHEN B_Date LIKE '11%' THEN '19' || SUBSTR(B_Date, 3, 6)
                                                          WHEN B_Date LIKE '13%' THEN '19' || SUBSTR(B_Date, 3, 6)
                                                          WHEN B_Date LIKE '10%' THEN '19' || SUBSTR(B_Date, 3, 6)
                                                          ELSE B_Date
                                                      END, 'YYYYMMDD')),
                     TO_DATE(CASE
                                  WHEN L_Date LIKE '11%' THEN '19' || SUBSTR(L_Date, 3, 6)
                                  WHEN L_Date LIKE '13%' THEN '19' || SUBSTR(L_Date, 3, 6)
                                  WHEN L_Date LIKE '10%' THEN '19' || SUBSTR(L_Date, 3, 6)
                                  ELSE L_Date
                              END, 'YYYYMMDD')))) / 365) AS C_Age
    FROM (
             SELECT Acct_Num,
                    Cust,
                    Categ,
                    Op_Date,
                    Arr_Id,
                    Acct_Off,
                    W_Bal
             FROM Randomized.Account_Info
             WHERE Acct_Num LIKE '01%'
               AND Cust != '999999'
             UNION
             SELECT Cont_Id,
                    Cust,
                    Categ,
                    Val_Date,
                    Draw_Acct,
                    Off,
                    Princ
             FROM Randomized.Money_Info
             WHERE Categ != '21037'
         ) A
              LEFT JOIN Randomized.Cust_Info B ON A.Cust = B.Recid
              LEFT JOIN Randomized.Arrangement_Info C ON A.Arr_Id = C.Arr_Id
              LEFT JOIN Randomized.alloc d ON A.Acct_Off = d.Code
    WHERE CASE
              WHEN Cat BETWEEN 3000 AND 3999 AND NVL(W_Bal, 0) = 0 THEN 0
              ELSE 1
          END != 0
          AND d.S_Unit IN ('AC', 'HNW', 'MM')
          AND A.Cust NOT IN ('999999', '1079146')
)
WHERE C_Age > 1 AND C_Age < 90;
