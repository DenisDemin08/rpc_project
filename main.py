from osgeo import gdal
gdal.UseExceptions()

from approx.rpc_model import RPC
import approx.forward as task1
import approx.inverse as task2
import approx.utm as task3
import approx.output_grid as task4
import approx.full_chain as task5
import approx.compare_gdal as task6
import approx.approximation as task7
from approx.chain import create_samples, plot_error_map, measure_timing, plot_timing, ReferenceChain
import numpy as np

def main():
    rpc = RPC('data/rpc.txt')
    height = rpc.HEIGHT_OFF #655

    #task1.run(rpc, height)
    #task2.run(rpc, height)
    utm_data = task3.run(rpc, height)
    grid_data = task4.run(rpc, height, utm_data['utm_to_wgs84'], utm_data['wgs84_to_utm'])
    task5.run(rpc, height, utm_data['utm_to_wgs84'], grid_data)
    #task6.run(rpc, height, utm_data['utm_to_wgs84'], grid_data)
    #task7.run_approximation(rpc, height, utm_data['utm_to_wgs84'], utm_data['wgs84_to_utm'], grid_data)

    # ЭТАП 2
    print("\n" + "="*60)
    print("Эталонная цепочка, выборки и время")
    print("="*60)
    
    # 1. Генерация выборок
    print("\nГенерация выборок...")
    samples_info = create_samples(rpc, height, utm_data['utm_to_wgs84'], grid_data, path="data/samples.npz")
    for name, st in samples_info.items():
        print(f"{name}: всего {st['total']}, внутри снимка {st['inside']} ({100.0 * st['frac']:.2f}%)")

    # 2. Карта ошибок (искусственная)
    print("\nПостроение карты ошибок...")
    samples = np.load("data/samples.npz")
    plot_error_map(samples["train_m_tr"], samples["train_n_tr"], samples["train_m_tr"] / 1000.0, "results/02_error_map_test.png")

    # 3. Замер времени
    print("\nЗамер времени (это займет около минуты)...")
    chain = ReferenceChain(rpc, height, utm_data['utm_to_wgs84'], grid_data['gt'])
    timing_results, step_info = measure_timing(chain, grid_data['width_tr'], grid_data['height_tr'])
    plot_timing(timing_results, "results/02_timing_chain.png")

    # 4. Вывод результатов
    t_1e6 = dict(timing_results)[10**6]
    time_per_point = t_1e6 / 1e6
    slow_names = ["аффинное", "UTM->B,L", "прямое RPC"]
    slow_name = slow_names[int(np.argmax([step_info['t1'], step_info['t2'], step_info['t3']]))]

    print(f"\nВремя на точку при N=10^6: {time_per_point:.3e} с/точка")
    print(f"Доли шагов: аффинное {step_info['fracs'][0]:.2f}%, UTM->B,L {step_info['fracs'][1]:.2f}%, RPC {step_info['fracs'][2]:.2f}%")
    print(f"Самый медленный шаг: {slow_name}")

    # 5. notes.md
    section = " Эталонная цепочка, выборки и время\n\nВыборки:\n"
    for name, st in samples_info.items():
        section += f"- `{name}`: {st['total']} точек, внутри снимка {st['inside']} ({100.0 * st['frac']:.2f}%).\n"
    section += f"\nВремя `reference_chain`:\n"
    section += f"- при N = 10^6: медиана {t_1e6:.3f} с, т.е. {time_per_point:.3e} с/точка;\n"
    section += f"- доли шагов при N = 10^6: аффинное {step_info['fracs'][0]:.2f}%, UTM->B,L {step_info['fracs'][1]:.2f}%, прямое RPC {step_info['fracs'][2]:.2f}%.\n\n"
    section += f"Самый медленный шаг: **{slow_name}**.\n"

    start_tag, end_tag = "<!-- STAGE2_START -->", "<!-- STAGE2_END -->"
    old_notes = ""
    try:
        with open("notes.md", "r", encoding="utf-8") as f:
            old_notes = f.read()
    except FileNotFoundError:
        pass

    if start_tag in old_notes and end_tag in old_notes:
        pre, post = old_notes.split(start_tag)[0], old_notes.split(end_tag)[1]
        new_notes = pre + start_tag + "\n" + section + "\n" + end_tag + post
    else:
        sep = "\n" if old_notes and not old_notes.endswith("\n") else ""
        new_notes = old_notes + sep + start_tag + "\n" + section + "\n" + end_tag + "\n"

    with open("notes.md", "w", encoding="utf-8") as f:
        f.write(new_notes)
    print("\nРезультаты сохранены в data/samples.npz, results/ и notes.md")

if __name__ == '__main__':
    main()