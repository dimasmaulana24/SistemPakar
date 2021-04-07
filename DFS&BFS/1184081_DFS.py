#contoh Peta
peta = {'A':set(['B']),
         'B':set(['A','C','D']),
         'C':set(['B','E']),
         'D':set(['B','G','H']),
         'E':set(['C','F','G']),
         'F':set(['E']),
         'G':set(['D','E','I','J']),
         'H':set(['D','J']),
  'I':set(['G'])}
    

def dfs(graf, mulai, tujuan):
    stack = [[mulai]]
    visited = set()

    while stack:     
        # masukkan antrian paling depan ke variabel jalur
        jalur = stack.pop(-1)

        # simpan node yang dipilih ke variabel state, misal jalur = C,B maka simpan B ke variabel state
        state = jalur[-1]

        # cek state apakah sama dengan tujuan, jika ya maka return jalur
        if state == tujuan:
            return jalur
        # jika state tidak sama dengan tujuan, maka cek apakah state tidak ada di visited
        elif state not in visited:
            # jika state tidak ada di visited maka cek cabang
            for cabang in graf.get(state, []): #cek semua cabang dari state
                jalur_baru = list(jalur) #masukkan isi dari variabel jalur ke variabel jalur_baru
                jalur_baru.append(cabang) #update/tambah isi jalur_baru dengan cabang
                stack.append(jalur_baru) #update/tambah queue dengan jalur_baru

            # tandai state yang sudah dikunjungi sebagai visited
            visited.add(state)

        #cek isi antrian
        isi = len(stack)
        if isi == 0:
            print("Tidak ditemukan")