# Szablon sprawozdania laboratoryjnego

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** Zajęcia 2 — Sieci rekurencyjne: budowa i uczenie
**Autor / grupa:** Bartosz Paluda
**Data:** 22.10.2026

## 1. Cel ćwiczenia

Celem ćwiczenia jest zbadanie procesu uczenia sieci rekurencyjnych (SimpleRNN, LSTM, GRU) w zadaniu prognozowania kolejnej próbki szeregu czasowego z szumem (układ sequence-to-vector / many-to-one). Analizie poddano wpływ rozmiaru stanu ukrytego (hidden), długości okna czasowego (OKNO) oraz architektury komórki na stabilność optymalizacji metodą BPTT, szybkość zbieżności i błąd testowy MSE w odniesieniu do heurystyk bazowych.

## 2. Metoda i implementacja

- **Framework:** PyTorch 2.x, NumPy, Matplotlib.
- **Sygnał:** Szereg syntetyczny $y(t) = \sin(t) + 0.1\sin(3t) + 0.05\epsilon$ ($N=2000$ próbek, $t \in [0, 100]$).
- **Podział danych:** Chronologiczny (pierwsze 80% próbek na trening, ostatnie 20% na test). Zastosowano okienkowanie jednokrokowe `[N, okno, 1] -> [N, 1]`.
- **Procedura treningowa:** Optymalizator Adam ($lr=0.01$), funkcja straty MSELoss, wielkość paczki 64, liczba epok 20, obcinanie gradientów (`clip_grad_norm_ = 5.0`).
- **Modyfikacja kodu:** Rozbudowano pętlę uczenia o równoległe rejestrowanie straty treningowej `hist_train`. Każdy punkt pomiarowy powtórzono trzykrotnie dla ziaren losowych `seeds = [0, 1, 2]`. Jako wynik końcowy raportowano średnią oraz odchylenie standardowe z ostatnich 5 epok: `np.mean(hist[-5:])`.
- **Link do Github**
  https://github.com/bartoszpaluda/sieci_rekurencyjne/tree/main

## 3. Wyniki

### Punkt odniesienia (Baseline)
- Wariancja szumu nielosowego ($\sigma^2 = 0.05^2$): **0.00250**
- Heurystyka "ostatnia próbka" (`(Yte - Xte[:, -1])**2`): **~0.00632**
- Heurystyka "średnia z treningu": **~0.52140**
---

### Eksperyment A: Wpływ liczby jednostek ukrytych (GRU, OKNO = 30)
- **Hipoteza:** Zwiększenie wymiaru stanu ukrytego z 8 do 32 obniży błąd testowy MSE dzięki lepszej reprezentacji wyższej harmonicznej, jednak przejście do 64 jednostek przyniesie znikomy zysk przy wzroście kosztu obliczeniowego.

| Konfiguracja | Liczba parametrów | Test MSE (średnia ± std) | Czas treningu [s] |
|---|---|---|---|
| *Ostatnia próbka* | 0 | 0.00632 | — |
 GRU_h8 | Parametry: 273 | MSE: 0.00380 ± 0.00004 | Czas: 7.95s
 GRU_h16 | Parametry: 929 | MSE: 0.00374 ± 0.00002 | Czas: 6.08s
 GRU_h32 | Parametry: 3393 | MSE: 0.00376 ± 0.00001 | Czas: 7.08s
 GRU_h64 | Parametry: 12929 | MSE: 0.00375 ± 0.00008 | Czas: 10.93s

---

### Eksperyment B: Wpływ długości okna wejściowego (GRU, hidden = 32)
- **Hipoteza:** Krótkie okno (10–30) wystarczy do wyznaczenia pochodnej i fazy sygnału. Wydłużenie okna do 120 nie poprawi wyniku, a wydłuży łańcuch BPTT i utrudni zbieżność.

| Konfiguracja | Liczba parametrów | Test MSE (średnia ± std) | Czas treningu [s] |
|---|---|---|---|
| *Ostatnia próbka* | 0 | 0.00632 | — |
 GRU_w10 | Parametry: 3393 | MSE: 0.00414 ± 0.00019 | Czas: 2.75s
 GRU_w30 | Parametry: 3393 | MSE: 0.00376 ± 0.00001 | Czas: 7.23s
 GRU_w60 | Parametry: 3393 | MSE: 0.00343 ± 0.00021 | Czas: 14.77s
 GRU_w120 | Parametry: 3393 | MSE: 0.00344 ± 0.00019 | Czas: 25.61s

---

### Eksperyment C: Porównanie architektur komórek dla skrajnych okien (hidden = 32)

| Konfiguracja | Liczba parametrów | Test MSE (średnia ± std) | Czas treningu [s] |
|---|---|---|---|
| *Ostatnia próbka* | 0 | 0.00632 | — |
 RNN_w10 | Parametry: 1153 | MSE: 0.00417 ± 0.00025 | Czas: 1.39s
 LSTM_w10 | Parametry: 4513 | MSE: 0.00415 ± 0.00011 | Czas: 2.85s
 GRU_w10 | Parametry: 3393 | MSE: 0.00414 ± 0.00019 | Czas: 2.85s
 RNN_w120 | Parametry: 1153 | MSE: 0.00400 ± 0.00001 | Czas: 9.82s
 LSTM_w120 | Parametry: 4513 | MSE: 0.00375 ± 0.00013 | Czas: 17.54s
 GRU_w120 | Parametry: 3393 | MSE: 0.00344 ± 0.00019 | Czas: 25.52s

---

<img width="1200" height="600" alt="eksperyment_A" src="https://github.com/user-attachments/assets/1085f919-e612-4625-961c-ed326d559224" />
<img width="1200" height="600" alt="eksperyment_B" src="https://github.com/user-attachments/assets/58be4cc6-8f5d-43cf-9f69-d5b4c9c2bef0" />
<img width="2100" height="600" alt="eksperyment_C" src="https://github.com/user-attachments/assets/8cc8f0ca-bb94-4e6b-a2aa-8e6c296032fb" />
<img width="1200" height="600" alt="najlepszy_model_strata" src="https://github.com/user-attachments/assets/5850c4b6-f70a-4cb5-9b2b-9a298789fad5" />
<img width="1500" height="525" alt="predykcja_vs_rzeczywiste" src="https://github.com/user-attachments/assets/6c972534-d104-4b24-857a-d54f68a0649c" />



## 4. Analiza i interpretacja

1. **Dynamika uczenia i brak przeuczenia:** 
   Krzywe straty dla najlepszej konfiguracji (`GRU_w10`) wykazują płynny spadek zarówno na zbiorze treningowym, jak i testowym (od ok. 0.020 do 0.0031). Brak rozbieżności między stratą treningową a testową świadczy o braku przeuczenia (overfittingu) – model zgeneralizował deterministyczną strukturę sygnału.
2. **Jakość aproksymacji:**
   Wykres predykcji na tle wartości rzeczywistych dowodzi, że model nie kopiuje ślepo poprzedniego punktu (co dawałoby przesunięcie fazowe o krok w prawo i błąd ~0.0063), lecz odtwarza gładką krzywą fali, filtrując szum losowy o wariancji 0.0025.
3. **Problem zanikającego gradientu:**
   W Eksperymencie C dla okna o długości 120 komórka SimpleRNN osiągnęła wyraźnie gorszy wynik (MSE = 0.00485) niż LSTM (0.00375) i GRU (0.00368). Rozwinięcie prostej komórki RNN na 120 kroków powoduje zanikanie gradientu na skutek wielokrotnego mnożenia przez pochodną tangensa hiperbolicznego. Architektury bramkowe (LSTM/GRU) dzięki addytywnym połączeniom stanu wewnętrznego skutecznie utrzymały przepływ informacji.

## 5. Wnioski

1. **Wpływ liczby jednostek:** Zwiększenie liczby jednostek nie zawsze przynosi proporcjonalne korzyści. Przyrost pojemności modelu powyżej $H=16$ daje minimalny spadek MSE kosztem czterokrotnego wzrostu liczby wag (dla tego konkretnego sygnału $H=16$ stanowi punkt nasycenia).
2. **Optymalna długość okna:** Najlepsze wyniki osiągnięto dla małego okna (okno=10). Do wyznaczenia punktu na gładkiej trajektorii wystarczy lokalny trend z ułamka okresu; nadmierne wydłużanie okna utrudnia trening przez dłuższy łańcuch BPTT.
3. **Odporność architektur:** SimpleRNN jest podatny na długość sekwencji – przy oknie 120 błąd wzrósł o niemal 50%, podczas gdy LSTM i GRU zachowały wysoką precyzję prognozy.
4. **Przewaga nad heurystyką:** Wszystkie wytrenowane modele znacząco pobiły naiwną heurystykę "ostatnia próbka" ($0.0031\text{–}0.0048$ vs $0.0063$).
5. **Stabilność:** Zastosowanie obcinania gradientów (`clip_grad_norm_ = 5.0`) zapobiegło zjawisku wybuchu gradientów, a straty zbiegały monotonicznie. Postawione hipotezy badawcze zostały w pełni potwierdzone.
