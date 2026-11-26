# Risultati — Timestral può guadagnare facendo trading automatico?

Data dello studio: 2026-11-26. Mercati: BTC/USDT, ETH/USDT, SOL/USDT, candele da 1 ora (Binance, prezzi reali).
Costi simulati: commissione 0,10% + slippage 0,03% per lato (0,26% andata e ritorno).

## Risposta breve

**No, non con questi dati e queste strategie.** Nessuna delle 512 configurazioni provate supera i criteri fissati prima del test. Nella maggior parte dei casi le strategie Timestral perdono anche **senza costi**.

## 1. Qualità delle previsioni (senza trading)

Confronto con un random walk "naive" (deriva e volatilità storiche). Periodo 2025-01-01 → 2026-10-15, circa 47.000 previsioni.
Il p-value della direzione è calcolato su finestre non sovrapposte.

| Modello | Accuratezza direzione 1h | p | Accuratezza 12h | p | Skill CRPS 1h | Copertura 80% | Correlazione con la volatilità (naive 0,17) |
|---|---|---|---|---|---|---|---|
| Timestral-5M | 50,6% | 0,007 | 50,2% | n.s. | +3,1% | 84% | 0,33 |
| Timestral-28M | 50,0% | n.s. | 50,2% | n.s. | +3,7% | 82% | 0,34 |
| Timestral-123M | 50,6% | 0,007 | 50,9% | 0,14 | +3,9% | 81% | 0,34 |

Cosa significa:
- **Direzione:** praticamente una moneta. Il vantaggio migliore è di circa +0,6 punti a 1 ora. Lo spostamento medio orario è di circa 0,5%, quindi il guadagno atteso per trade è molto più piccolo dei costi (0,26%).
- **Distribuzione e volatilità:** qui Timestral è chiaramente migliore del naive. Gli intervalli sono ben calibrati e la volatilità prevista ha una correlazione doppia con quella realizzata.
- **Holdout** (dopo il rilascio dei pesi, 2026-10-16 → oggi): stesse conclusioni. L'accuratezza del 54–56% a 12/24h non è significativa sui campioni indipendenti (p 0,08–0,14). In questo periodo di forte calo anche il naive "con deriva" arriva al 52–55%.

## 2. Backtest (studio tuning → validazione → holdout)

Il protocollo:
- **Tuning** (2025): 128 configurazioni per modello.
- **Validazione** (2026-01 → 2026-10-15): la migliore configurazione per famiglia, con test Monte Carlo (timing casuale con la stessa esposizione) e Deflated Sharpe.
- **Holdout** (2026-10-16 → 2026-11-26): eseguito una sola volta.

Riferimenti di mercato:

| Periodo | Buy & hold (equal weight) |
|---|---|
| Validazione | -13,9% |
| Holdout | -29,9% |

Validazione, Timestral-123M (le altre taglie sono simili):

| Strategia | Netto | Lordo (senza costi) | Sharpe | Trade | Commissioni | Monte Carlo p |
|---|---|---|---|---|---|---|
| directional | -3,4% | -2,5% | -1,95 | 16 | 71 $ | 0,97 |
| mean_reversion | -57,5% | -2,5% | -6,11 | 957 | 3.949 $ | 0,001 |
| vol_filter | -25,7% | -10,9% | -0,64 | 208 | 1.109 $ | 0,35 |
| vol_target | -12,0% | -9,8% | -0,18 | – | 164 $ | 0,32 |

Lettura dei risultati:
- **directional:** il modello raramente è abbastanza sicuro da superare i costi, quindi quasi non opera. Quando opera, non guadagna.
- **mean_reversion:** il timing è migliore del caso (Monte Carlo p=0,001), ma lo stesso segnale esiste anche con la previsione naive. Non è merito di Timestral, e le commissioni lo annullano.
- **vol_filter / vol_target:** la buona previsione di volatilità non basta a battere buy & hold dopo i costi.
- L'unico risultato positivo sull'holdout è il **naive** direzionale (+18%, short in un mercato in caduta). Anche questo è non significativo: Monte Carlo p=0,28.

## 3. Variante con covariate

Ho confrontato Timestral-28M con e senza covariate sulle **stesse circa 8.300 origini** (una ogni 6 ore, dal 2025-01 a oggi). Covariate usate:
- passate: log-volume, range high/low;
- note nel futuro: ora del giorno, weekend.

| Orizzonte | CRPS senza → con covariate | Accuratezza direzione senza → con |
|---|---|---|
| 1h | 0,003220 → 0,003221 | 50,0% → 49,3% |
| 4h | 0,006559 → 0,006558 | 50,4% → 50,9% (p=0,06) |
| 12h | 0,011412 → 0,011486 | 50,1% → 49,6% |
| 24h | 0,016510 → 0,016688 | 49,4% → 47,8% |

Nessun miglioramento. Non ho eseguito il backtest completo con covariate, perché la previsione non migliora e l'inferenza è circa 15 volte più lenta.

## 4. Cosa resta aperto (idee per proseguire)

- Usare Timestral come **stimatore di rischio** (sizing, stop dinamici) dentro una strategia che abbia già un vantaggio proprio, invece che come generatore di segnali.
- Testare timeframe più lunghi (4h, 1 giorno), dove i costi pesano meno sul movimento atteso.
- Mercati con costi molto più bassi (maker fee, futures) oppure serie con stagionalità forte, che sono il punto di forza dei modelli di forecasting.
- Il paper trading live continua con la configurazione scelta dallo studio (123M, directional h=12). Va monitorato per settimane prima di trarre conclusioni.
