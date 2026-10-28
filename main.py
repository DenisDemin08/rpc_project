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

def main():
    rpc = RPC('data/rpc.txt')
    height = rpc.HEIGHT_OFF #655

    task1.run(rpc, height)
    task2.run(rpc, height)
    utm_data = task3.run(rpc, height)
    grid_data = task4.run(rpc, height, utm_data['utm_to_wgs84'], utm_data['wgs84_to_utm'])
    task5.run(rpc, height, utm_data['utm_to_wgs84'], grid_data)
    task6.run(rpc, height, utm_data['utm_to_wgs84'], grid_data)
    task7.run_approximation(rpc, height, utm_data['utm_to_wgs84'], utm_data['wgs84_to_utm'], grid_data)

if __name__ == '__main__':
    main()