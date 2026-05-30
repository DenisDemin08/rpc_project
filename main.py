from osgeo import gdal
gdal.UseExceptions()

from rpc_model import RPC
import forward as task1
import inverse as task2
import utm as task3
import output_grid as task4
import full_chain as task5
import compare_gdal as task6

def main():
    rpc = RPC('rpc.txt')
    height = rpc.HEIGHT_OFF   # 655 м

    task1.run(rpc, height)
    task2.run(rpc, height)
    utm_data = task3.run(rpc, height)
    grid_data = task4.run(rpc, height, utm_data['utm_to_wgs84'], utm_data['wgs84_to_utm'])
    task5.run(rpc, height, utm_data['utm_to_wgs84'], grid_data)
    task6.run(rpc, height, utm_data['utm_to_wgs84'], grid_data)

if __name__ == '__main__':
    main()